/**
 *  Copyright (c) 2015 by Contributors
 */
#include "ps/internal/van.h"
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <limits>
#include "ps/base.h"
#include "ps/sarray.h"
#include "ps/internal/postoffice.h"
#include "ps/internal/customer.h"
#include "./network_utils.h"
#include "./meta.pb.h"
#include "./zmq_van.h"
#include "./resender.h"
#include <dmlc/logging.h>

namespace ps {

// interval in second between to heartbeast signals. 0 means no heartbeat.
// don't send heartbeast in default. because if the scheduler received a
// heartbeart signal from a node before connected to that node, then it could be
// problem.
const static int kDefaultHeartbeatInterval = 0;

Van* Van::Create(const std::string& type) {
  if (type == "zmq") {
    return new ZMQVan();
  } else {
    LOG(FATAL) << "unsupported van type: " << type;
    return nullptr;
  }
}

void Van::Start() {
  // get scheduler info
  scheduler_.hostname = std::string(CHECK_NOTNULL(Environment::Get()->find("DMLC_PS_ROOT_URI")));
  scheduler_.port     = atoi(CHECK_NOTNULL(Environment::Get()->find("DMLC_PS_ROOT_PORT")));
  scheduler_.role     = Node::SCHEDULER;
  scheduler_.id       = kScheduler;
  is_scheduler_       = Postoffice::Get()->is_scheduler();

  // get my node info
  if (is_scheduler_) {
    my_node_ = scheduler_;
  } else {
    auto role = is_scheduler_ ? Node::SCHEDULER :
                (Postoffice::Get()->is_worker() ? Node::WORKER : Node::SERVER);
    const char* nhost = Environment::Get()->find("DMLC_NODE_HOST");
    std::string ip;
    if (nhost) ip = std::string(nhost);
    if (ip.empty()) {
      const char*  itf = Environment::Get()->find("DMLC_INTERFACE");
      std::string interface;
      if (itf) interface = std::string(itf);
      if (interface.size()) {
        GetIP(interface, &ip);
      } else {
        GetAvailableInterfaceAndIP(&interface, &ip);
      }
      CHECK(!interface.empty()) << "failed to get the interface";
    }
    int port = GetAvailablePort();
    const char* pstr = Environment::Get()->find("PORT");
    if (pstr) port = atoi(pstr);
    CHECK(!ip.empty()) << "failed to get ip";
    CHECK(port) << "failed to get a port";
    my_node_.hostname = ip;
    my_node_.role     = role;
    my_node_.port     = port;
    // cannot determine my id now, the scheduler will assign it later
    // set it explicitly to make re-register within a same process possible
    my_node_.id = Node::kEmpty;
  }

  // bind.
  my_node_.port = Bind(my_node_, is_scheduler_ ? 0 : 40);
  PS_VLOG(1) << "Bind to " << my_node_.DebugString();
  CHECK_NE(my_node_.port, -1) << "bind failed";

  // connect to the scheduler
  Connect(scheduler_);

  // for debug use
  if (Environment::Get()->find("PS_DROP_MSG")) {
    drop_rate_ = atoi(Environment::Get()->find("PS_DROP_MSG"));
  }
  // start receiver
  receiver_thread_ = std::unique_ptr<std::thread>(
      new std::thread(&Van::Receiving, this));

  if (!is_scheduler_) {
    // let the scheduler know myself
    Message msg;
    msg.meta.recver = kScheduler;
    msg.meta.control.cmd = Control::ADD_NODE;
    msg.meta.control.node.push_back(my_node_);
    msg.meta.timestamp = timestamp_++;
    Send(msg);
  }
  // wait until ready
  while (!ready_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // resender
  if (Environment::Get()->find("PS_RESEND") && atoi(Environment::Get()->find("PS_RESEND")) != 0) {
    int timeout = 1000;
    if (Environment::Get()->find("PS_RESEND_TIMEOUT")) {
      timeout = atoi(Environment::Get()->find("PS_RESEND_TIMEOUT"));
    }
    resender_ = new Resender(timeout, 10, this);
  }

  if (!is_scheduler_) {
    // start heartbeat thread
    heartbeat_thread_ = std::unique_ptr<std::thread>(
      new std::thread(&Van::Heartbeat, this));
  }

  if (is_scheduler_) {
    task_queue_ = std::unique_ptr<std::priority_queue<TaskSpec>>(
      new std::priority_queue<TaskSpec>()
    );
    // start cancel thread on the scheduler
    cancel_thread_ = std::unique_ptr<std::thread>(
      new std::thread(&Van::ExecuteTasks, this)
    );
  }
}

void Van::Stop() {
  // stop threads
  Message exit;
  exit.meta.control.cmd = Control::TERMINATE;
  exit.meta.recver = my_node_.id;
  SendMsg(exit);
  receiver_thread_->join();
  if (!is_scheduler_) heartbeat_thread_->join();
  if (is_scheduler_) cancel_thread_->join();
  if (resender_) delete resender_;
}

int Van::Send(const Message& msg) {
  int send_bytes = SendMsg(msg);
  CHECK_NE(send_bytes, -1);
  send_bytes_ += send_bytes;
  if (resender_) resender_->AddOutgoing(msg);
  if (Postoffice::Get()->verbose() >= 2) {
    PS_VLOG(2) << msg.DebugString();
  }
  return send_bytes;
}

#define TO_MS(x) std::chrono::duration_cast<std::chrono::milliseconds>(x)
#define MS_TO_DOUBLE(x) TO_MS(x).count()*1.0

void Van::OnRecvReport(Message &msg) {
  std::lock_guard<std::mutex> lock(cancel_mutex_);

  if (epoch_num_ >= 1 && epoch_num_ <= 41) {
    auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    int id = Postoffice::IDtoRank(msg.meta.sender);

    // read loss from message
    std::istringstream iss(msg.meta.body);
    double loss;
    iss >> loss;
    // LOG(INFO) << "Received loss " << loss << " from " << id;

    loss_trace_->at(id).push_back({loss, time});
  }
}

std::chrono::milliseconds Van::GetTrialWaitTime(int epoch_num) {
  double percent = (8 - ((epoch_num - 1) / 5)) / 8.0;
  return TO_MS(percent * upper_ratio_ * epoch_length_);
}

void Van::UpdateThreshold() {
  int count = 0;
  int num = 0;
  // LOG(INFO) << "size: " << push_trace_->size();
  for (int i = 0; i < push_trace_->size() - 1; i++) {
    // LOG(INFO) << "i: " << i;
    if (push_trace_->at(i + 1) - push_trace_->at(i) > wait_time_ &&
        push_trace_->at(i) - push_trace_->at(0) > wait_time_) {
      num += 1;
      for (int j = i - 1; j >= 0; j--) {
        // LOG(INFO) << "j: " << j;
        if (push_trace_->at(i) - push_trace_->at(j) <= wait_time_) {
          count += 1;
        }
      }
    }
  }
  push_trace_->clear();
  if (num != 0) {
    double new_threshold = (cancel_threshold_ * epoch_num_ + 1.0 * count / num) / (epoch_num_ + 1);
    LOG(INFO) << "Update threshold from " << cancel_threshold_ << " to " << new_threshold << ".";
    cancel_threshold_ = new_threshold;
  }
}

void Van::SetOptimalWaitTime() {
  for (int k = 0; k < num_workers_; k++) {
    LOG(INFO) << "Worker " << k << " got " << loss_trace_->at(k).size() << " losses.";
  }
  double msd = 0; // minimal second order derivative
  int min = 6;    // index of minimum (6 is default setting)
  // calculate second order derivative for every trial
  for (int i = 1; i < 8; i++) {
    LOG(INFO) << "In the " << i << "-th trial";
    // each trial lasts 5 epochs
    std::time_t time[5]; // average endtime for each epoch
    double loss[5];      // average loss for each epoch
    for (int j = 0; j < 5; j++) {
      std::time_t time_sum = 0;
      double loss_sum = 0;
      int data_num = 0;
      for (int k = 0; k < num_workers_; k++) { // average over all workers
        int index = i * 5 + j;
        // the loss may not be available yet
        if (loss_trace_->at(k).size() > index) {
          loss_sum += loss_trace_->at(k)[index].first;
          time_sum += loss_trace_->at(k)[index].second;
          data_num += 1;
        }
      }
      loss[j] = loss_sum / data_num;
      time[j] = time_sum / data_num;
      LOG(INFO) << "loss[" << j << "]: " << loss[j] << " time[" << j << "]:" << time[j];
    }
    // calculate first order derivative
    for (int j = 0; j < 4; j++) {
      loss[j] = (loss[j+1] - loss[j]) / (time[j+1] - time[j]);
      time[j] = (time[j+1] + time[j]) / 2;
    }
    // calculate sum of second order derivative
    double sd_sum = 0;
    for (int j = 0; j < 3; j++) {
      sd_sum += (loss[j+1] - loss[j]) / (time[j+1] - time[j]);
    }
    LOG(INFO) << "Second order derivative is " << sd_sum / 3 << " in the " << i << "-th trial.";
    // check whether it is the maximum
    if (sd_sum < msd) {
      min = i;
      msd = sd_sum;
    }
  }
  wait_time_ = GetTrialWaitTime(min * 5 + 1);
  cancel_threshold_ = (8 - min) / 8.0 * upper_ratio_ * num_workers_;
}

/**
 * 1. enqueue worker into active search list
 * 2. increment the number of expected updates for all workers in active search list
 * 3. find wait_time of optimal expectation
 *
 * @param msg received message
 */
void Van::OnRecvNotif(Message &msg) {
  std::lock_guard<std::mutex> lock(cancel_mutex_);
  if (push_trace_ == nullptr) {
    InitializeCancellation();
  }

  auto time = std::chrono::system_clock::now();
  int id = Postoffice::IDtoRank(msg.meta.sender);

  if (push_num_ > 0 && push_num_ % num_workers_ == 0) {

    // record epoch length
    auto cur_epoch_length = TO_MS(time - push_trace_->at(0));
    epoch_length_ = (epoch_length_ * epoch_num_ + cur_epoch_length) / (epoch_num_ + 1);
    epoch_num_ += 1;

    // dynamically adapt wait_time_ and cancel_threshold_
    if (epoch_num_ < 41) {
      wait_time_ = GetTrialWaitTime(epoch_num_);
      LOG(INFO) << "Trial wait_time for next epoch is " << MS_TO_DOUBLE(wait_time_) << ".";
    } else if (epoch_num_ == 42) {
      SetOptimalWaitTime();
      LOG(INFO) << "Set wait_time to " << MS_TO_DOUBLE(wait_time_) << " and cancel_threshold_ to " << cancel_threshold_ << ".";
    } else if (epoch_num_ > 42) {
      UpdateThreshold();
    }
    
    push_trace_->clear();
  }

  push_trace_->push_back(time);
  push_num_ += 1;

  // increment update count for all waiting workers
  for (auto worker : *waiting_workers_) {
    update_count_->at(worker) += 1;
  }

  if (MS_TO_DOUBLE(wait_time_) > 0) {
    // add current worker to waiting_workers_
    update_count_->at(id) = 0;
    waiting_workers_->insert(id);
    task_queue_->push({ CHECK, id, time + wait_time_ });
  }
}

void Van::InitializeCancellation() {
  push_num_ = 0;
  epoch_num_ = 0;
  cancel_threshold_ = 1;
  upper_ratio_ = std::sqrt(2 * (num_workers_ * num_workers_ + 1)) / (num_workers_ + 1) - 1;
  push_trace_ = std::unique_ptr<TimeVector>(
    new TimeVector(num_workers_)
  );
  push_trace_->clear();
  waiting_workers_ = std::unique_ptr<std::set<int>>(
    new std::set<int>()
  );
  update_count_ = std::unique_ptr<std::vector<int>>(
    new std::vector<int>(num_workers_, 0)
  );
  loss_trace_ = std::unique_ptr<std::vector<LossVector>>(
    new std::vector<LossVector>(num_workers_)
  );
}

/**
 * 1. dequeue worker from active search list
 * 2. send cancellation if the condition is met
 * 3. start timer callback for next worker
 */
void Van::ExecuteTasks() {
  while (true) {
    std::chrono::system_clock::time_point next_wake;
    {
      std::lock_guard<std::mutex> lock(cancel_mutex_);
      if (!task_queue_->empty()) {
        auto spec = task_queue_->top();
        task_queue_->pop();
        switch (spec.task) {
          case CHECK:
            CheckCancellation(spec.id, spec.time);
            break;
        }
      }
      if (task_queue_->empty()) {
        std::chrono::milliseconds one_sec { 1000 };
        next_wake = std::chrono::system_clock::now() + one_sec;
      } else {
        next_wake = task_queue_->top().time;
      }
    } // scope for lock_guard
    std::this_thread::sleep_until(next_wake);
  }
}

void Van::CheckCancellation(int id, std::chrono::system_clock::time_point &now) {
  auto updates = update_count_->at(id);
  LOG(INFO) << "Worker " << id << " got " << updates << " updates, expected " << cancel_threshold_ << ".";
  if (updates >= std::round(cancel_threshold_)) {
    // issue cancellation
    Message msg;
    msg.meta.control.cmd = Control::CANCEL;
    msg.meta.customer_id = 0; // KVWorker will always be initialized with app_id set to 0
    msg.meta.request = false;
    msg.meta.push = false;
    msg.meta.recver = Postoffice::WorkerRankToID(id);
    Send(msg);
    // LOG(INFO) << "Issue cancellation to worker " << id << ".";
  }
  update_count_->at(id) = 0;
  auto pos = waiting_workers_->find(id);
  if (pos != waiting_workers_->end()) {
  	waiting_workers_->erase(pos);
  }
}

void Van::Receiving() {
  const char* heartbeat_timeout_val = Environment::Get()->find("PS_HEARTBEAT_TIMEOUT");
  const int heartbeat_timeout = heartbeat_timeout_val ? atoi(heartbeat_timeout_val) : kDefaultHeartbeatInterval;
  Meta nodes;  // for scheduler usage
  while (true) {
    Message msg;
    int recv_bytes = RecvMsg(&msg);

    // For debug, drop received message
    if (ready_ && drop_rate_ > 0) {
      unsigned seed = time(NULL) + my_node_.id;
      if (rand_r(&seed) % 100 < drop_rate_) {
        LOG(WARNING) << "Drop message " << msg.DebugString();
        continue;
      }
    }

    CHECK_NE(recv_bytes, -1);
    recv_bytes_ += recv_bytes;
    if (Postoffice::Get()->verbose() >= 2) {
      PS_VLOG(2) << msg.DebugString();
    }
    // duplicated message
    if (resender_ && resender_->AddIncomming(msg)) continue;

    if (!msg.meta.control.empty()) {
      // do some management
      auto& ctrl = msg.meta.control;
      if (ctrl.cmd == Control::TERMINATE) {
        PS_VLOG(1) << my_node_.ShortDebugString() << " is stopped";
        ready_ = false;
        break;
      } else if (ctrl.cmd == Control::ADD_NODE) {
        size_t num_nodes = Postoffice::Get()->num_servers() +
                           Postoffice::Get()->num_workers();
        auto dead_nodes = Postoffice::Get()->GetDeadNodes(heartbeat_timeout);
        std::unordered_set<int> dead_set(dead_nodes.begin(), dead_nodes.end());
        Meta recovery_nodes;  // store recovery nodes
        recovery_nodes.control.cmd = Control::ADD_NODE;
        // assign an id
        if (msg.meta.sender == Meta::kEmpty) {
          CHECK(is_scheduler_);
          CHECK_EQ(ctrl.node.size(), 1);
          if (nodes.control.node.size() < num_nodes) {
            nodes.control.node.push_back(ctrl.node[0]);
          } else {
            // some node dies and restarts
            CHECK(ready_);
            for (size_t i = 0; i < nodes.control.node.size() - 1; ++i) {
              const auto& node = nodes.control.node[i];
              if (dead_set.find(node.id) != dead_set.end() && node.role == ctrl.node[0].role) {
                auto& recovery_node = ctrl.node[0];
                // assign previous node id
                recovery_node.id = node.id;
                recovery_node.is_recovery = true;
                PS_VLOG(1) << "replace dead node " << node.DebugString()
                           << " by node " << recovery_node.DebugString();
                nodes.control.node[i] = recovery_node;
                recovery_nodes.control.node.push_back(recovery_node);
                break;
              }
            }
          }
        }

        // update my id
        for (size_t i = 0; i < ctrl.node.size(); ++i) {
          const auto& node = ctrl.node[i];
          if (my_node_.hostname == node.hostname &&
              my_node_.port == node.port) {
            my_node_ = node;
            std::string rank = std::to_string(Postoffice::IDtoRank(node.id));
#ifdef _MSC_VER
            _putenv_s("DMLC_RANK", rank.c_str());
#else
            setenv("DMLC_RANK", rank.c_str(), true);
#endif
          }
        }

        if (is_scheduler_) {
          time_t t = time(NULL);
          if (nodes.control.node.size() == num_nodes) {
            // sort the nodes according their ip and port,
            std::sort(nodes.control.node.begin(), nodes.control.node.end(),
                      [](const Node& a, const Node& b) {
                        return (a.hostname.compare(b.hostname) | (a.port < b.port)) > 0;
                      });
            // assign node rank
            for (auto& node : nodes.control.node) {
              CHECK_EQ(node.id, Node::kEmpty);
              int id = node.role == Node::SERVER ?
                       Postoffice::ServerRankToID(num_servers_) :
                       Postoffice::WorkerRankToID(num_workers_);
              PS_VLOG(1) << "assign rank=" << id << " to node " << node.DebugString();
              node.id = id;
              Connect(node);
              if (node.role == Node::SERVER) ++num_servers_;
              if (node.role == Node::WORKER) ++num_workers_;
              Postoffice::Get()->UpdateHeartbeat(node.id, t);
            }
            nodes.control.node.push_back(my_node_);
            nodes.control.cmd = Control::ADD_NODE;
            Message back; back.meta = nodes;
            for (int r : Postoffice::Get()->GetNodeIDs(
                     kWorkerGroup + kServerGroup)) {
              back.meta.recver = r;
              back.meta.timestamp = timestamp_++;
              Send(back);
            }
            PS_VLOG(1) << "the scheduler is connected to "
                    << num_workers_ << " workers and " << num_servers_ << " servers";
            ready_ = true;
          } else if (recovery_nodes.control.node.size() > 0) {
            // send back the recovery node
            CHECK_EQ(recovery_nodes.control.node.size(), 1);
            Connect(recovery_nodes.control.node[0]);
            Postoffice::Get()->UpdateHeartbeat(recovery_nodes.control.node[0].id, t);
            Message back;
            for (int r : Postoffice::Get()->GetNodeIDs(
                     kWorkerGroup + kServerGroup)) {
              if (r != recovery_nodes.control.node[0].id
                    && dead_set.find(r) != dead_set.end()) {
                // do not try to send anything to dead node
                continue;
              }
              // only send recovery_node to nodes already exist
              // but send all nodes to the recovery_node
              back.meta = (r == recovery_nodes.control.node[0].id) ? nodes : recovery_nodes;
              back.meta.recver = r;
              back.meta.timestamp = timestamp_++;
              Send(back);
            }
          }
        } else {
          for (const auto& node : ctrl.node) {
            Connect(node);
            if (!node.is_recovery && node.role == Node::SERVER) ++num_servers_;
            if (!node.is_recovery && node.role == Node::WORKER) ++num_workers_;
          }
          PS_VLOG(1) << my_node_.ShortDebugString() << " is connected to others";
          ready_ = true;
        }
      } else if (ctrl.cmd == Control::BARRIER) {
        if (msg.meta.request) {
          if (barrier_count_.empty()) {
            barrier_count_.resize(8, 0);
          }
          int group = ctrl.barrier_group;
          ++barrier_count_[group];
          PS_VLOG(1) << "Barrier count for " << group << " : " << barrier_count_[group];
          if (barrier_count_[group] ==
              static_cast<int>(Postoffice::Get()->GetNodeIDs(group).size())) {
            barrier_count_[group] = 0;
            Message res;
            res.meta.request = false;
            res.meta.control.cmd = Control::BARRIER;
            for (int r : Postoffice::Get()->GetNodeIDs(group)) {
              res.meta.recver = r;
              res.meta.timestamp = timestamp_++;
              CHECK_GT(Send(res), 0);
            }
          }
        } else {
          Postoffice::Get()->Manage(msg);
        }
      } else if (ctrl.cmd == Control::HEARTBEAT) {
        time_t t = time(NULL);
        for (auto &node : ctrl.node) {
          Postoffice::Get()->UpdateHeartbeat(node.id, t);
          if (is_scheduler_) {
            Message heartbeat_ack;
            heartbeat_ack.meta.recver = node.id;
            heartbeat_ack.meta.control.cmd = Control::HEARTBEAT;
            heartbeat_ack.meta.control.node.push_back(my_node_);
            heartbeat_ack.meta.timestamp = timestamp_++;
            // send back heartbeat
            Send(heartbeat_ack);
          }
        }
      } else if (ctrl.cmd == Control::NOTIFY) {
        // The receiver of NOTIFY message must be a scheduler.
        if (is_scheduler_) {
          OnRecvNotif(msg);
        }
      } else if (ctrl.cmd == Control::REPORT) {
        // The receiver of REPORT message must be a scheduler.
        if (is_scheduler_) {
          OnRecvReport(msg);
        }
      } else if (ctrl.cmd == Control::CANCEL) {
        if (Postoffice::Get()->is_worker()) {
          CHECK_NE(msg.meta.customer_id, Meta::kEmpty);
          int id = msg.meta.customer_id;
          auto *obj = Postoffice::Get()->GetCustomer(id, 5);
          obj->Accept(msg);
        }
      }
    } else {
      CHECK_NE(msg.meta.sender, Meta::kEmpty);
      CHECK_NE(msg.meta.recver, Meta::kEmpty);
      CHECK_NE(msg.meta.customer_id, Meta::kEmpty);
      int id = msg.meta.customer_id;
      auto* obj = Postoffice::Get()->GetCustomer(id, 5);
      CHECK(obj) << "timeout (5 sec) to wait App " << id << " ready";
      obj->Accept(msg);
    }
  }
}

void Van::PackMeta(const Meta& meta, char** meta_buf, int* buf_size) {
  // convert into protobuf
  PBMeta pb;
  pb.set_head(meta.head);
  if (meta.customer_id != Meta::kEmpty) pb.set_customer_id(meta.customer_id);
  if (meta.timestamp != Meta::kEmpty) pb.set_timestamp(meta.timestamp);
  if (meta.body.size()) pb.set_body(meta.body);
  pb.set_push(meta.push);
  pb.set_request(meta.request);
  pb.set_simple_app(meta.simple_app);
  for (auto d : meta.data_type) pb.add_data_type(d);
  if (!meta.control.empty()) {
    auto ctrl = pb.mutable_control();
    ctrl->set_cmd(meta.control.cmd);
    if (meta.control.cmd == Control::BARRIER) {
      ctrl->set_barrier_group(meta.control.barrier_group);
    } else if (meta.control.cmd == Control::ACK) {
      ctrl->set_msg_sig(meta.control.msg_sig);
    }
    for (const auto& n : meta.control.node) {
      auto p = ctrl->add_node();
      p->set_id(n.id);
      p->set_role(n.role);
      p->set_port(n.port);
      p->set_hostname(n.hostname);
      p->set_is_recovery(n.is_recovery);
    }
  }

  // to string
  *buf_size = pb.ByteSize();
  *meta_buf = new char[*buf_size+1];
  CHECK(pb.SerializeToArray(*meta_buf, *buf_size))
      << "failed to serialize protbuf";
}

void Van::UnpackMeta(const char* meta_buf, int buf_size, Meta* meta) {
  // to protobuf
  PBMeta pb;
  CHECK(pb.ParseFromArray(meta_buf, buf_size))
      << "failed to parse string into protobuf";

  // to meta
  meta->head = pb.head();
  meta->customer_id = pb.has_customer_id() ? pb.customer_id() : Meta::kEmpty;
  meta->timestamp = pb.has_timestamp() ? pb.timestamp() : Meta::kEmpty;
  meta->request = pb.request();
  meta->push = pb.push();
  meta->simple_app = pb.simple_app();
  meta->body = pb.body();
  meta->data_type.resize(pb.data_type_size());
  for (int i = 0; i < pb.data_type_size(); ++i) {
    meta->data_type[i] = static_cast<DataType>(pb.data_type(i));
  }
  if (pb.has_control()) {
    const auto& ctrl = pb.control();
    meta->control.cmd = static_cast<Control::Command>(ctrl.cmd());
    meta->control.barrier_group = ctrl.barrier_group();
    meta->control.msg_sig = ctrl.msg_sig();
    for (int i = 0; i < ctrl.node_size(); ++i) {
      const auto& p = ctrl.node(i);
      Node n;
      n.role = static_cast<Node::Role>(p.role());
      n.port = p.port();
      n.hostname = p.hostname();
      n.id = p.has_id() ? p.id() : Node::kEmpty;
      n.is_recovery = p.is_recovery();
      meta->control.node.push_back(n);
    }
  } else {
    meta->control.cmd = Control::EMPTY;
  }
}

void Van::Heartbeat() {
  const char* val = Environment::Get()->find("PS_HEARTBEAT_INTERVAL");
  const int interval = val ? atoi(val) : kDefaultHeartbeatInterval;
  while (interval > 0 && ready_) {
    std::this_thread::sleep_for(std::chrono::seconds(interval));
    Message msg;
    msg.meta.recver = kScheduler;
    msg.meta.control.cmd = Control::HEARTBEAT;
    msg.meta.control.node.push_back(my_node_);
    msg.meta.timestamp = timestamp_++;
    Send(msg);
  }
}
}  // namespace ps
