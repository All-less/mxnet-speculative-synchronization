/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_KV_APP_H_
#define PS_KV_APP_H_
#include <algorithm>
#include <utility>
#include <vector>
#include <thread>
#include <chrono>
#include <memory>
#include <ctime>
#include <map>
#include <mutex>
#include <list>
#include <sstream>

#include "ps/base.h"
#include "ps/simple_app.h"
#include "ps/internal/message.h"
#include <dmlc/logging.h>

namespace ps {

/**
 * \brief the structure for a list of key-value pairs
 *
 * The keys must be unique and sorted in an increasing order.  The length of a
 * value can be more than one. If \a lens is empty, then the length
 * of a value is determined by `k=vals.size()/keys.size()`.  The \a i-th KV pair
 * is then
 *
 * \verbatim {keys[i], (vals[i*k], ..., vals[(i+1)*k-1])} \endverbatim
 *
 * If \a lens is given, then `lens[i]` is the length of the \a i-th
 * value. Let
 *
 * \verbatim n = lens[0] + .. + lens[i-1]  \endverbatim
 *
 * then the \a i-th KV pair is presented as
 *
 * \verbatim {keys[i], (vals[n], ..., vals[lens[i]+n-1])} \endverbatim
 */
template <typename Val>
struct KVPairs {
  // /** \brief empty constructor */
  // KVPairs() {}
  /** \brief the list of keys */
  SArray<Key> keys;
  /** \brief the according values */
  SArray<Val> vals;
  /** \brief the according value lengths (could be empty) */
  SArray<int> lens;
};

/**
 * \brief type of cancellation callback
 */
typedef void (*CancelCallback)(void);

/**
 * \brief A worker node that can \ref Push (\ref Pull) key-value pairs to (from) server
 * nodes
 *
 * \tparam Val the type of value, which should be primitive types such as
 * int32_t and float
 */
template<typename Val>
class KVWorker : public SimpleApp {
 public:

  /**
   * \brief a map storing notification from servers,
   *        whose entry means server_id -> timestamp_of_last_notification
   */
  typedef std::map<int, double> NotificationMap;

  /** avoid too many this-> */
  using SimpleApp::obj_;
  /**
   * \brief callback function for \ref Push and \ref Pull
   *
   * It is called by the data receiving thread of this instance when the push or
   * pull is actually finished. Namely the kv pairs have already written into
   * servers' data structure or the kv pairs have already pulled back.
   */
  using Callback = std::function<void()>;

  /**
   * \brief constructor
   *
   * \param app_id the app id, should match with \ref KVServer's id
   */
  explicit KVWorker(int app_id) :
    SimpleApp(),
    notification_record_(std::unique_ptr<NotificationMap>(new NotificationMap))
  {
    using namespace std::placeholders;
    slicer_ = std::bind(&KVWorker<Val>::DefaultSlicer, this, _1, _2, _3);
    obj_ = new Customer(app_id, std::bind(&KVWorker<Val>::Process, this, _1));
    MXNET_PROFILING = GetEnv("MXNET_PROFILING", 0);
    MXNET_ENABLE_CANCEL = GetEnv("MXNET_ENABLE_CANCEL", 0);
    char *window = getenv("MXNET_CANCEL_WINDOW");
    MXNET_CANCEL_WINDOW = window ? atof(window) : 1.0;
    char *percent = getenv("MXNET_CANCEL_PERCENT");
    MXNET_CANCEL_PERCENT = percent ? atof(percent) / 100.0 : 0.1;
  }

  /** \brief deconstructor */
  virtual ~KVWorker() { delete obj_; obj_ = nullptr; }

  /**
   * \brief Pushes a list of key-value pairs to all server nodes.
   *
   * This function pushes a KV list specified by \a keys and \a vals to all
   * server nodes.
   *
   * Sample usage: the following codes push two KV pairs `{1, (1.1, 1.2)}` and `{3,
   * (3.1,3.2)}` to server nodes, where the value is a length-2 float vector
   * \code
   *   KVWorker<float> w;
   *   std::vector<Key> keys = {1, 3};
   *   std::vector<float> vals = {1.1, 1.2, 3.1, 3.2};
   *   w.Push(keys, vals);
   * \endcode
   *
   * If \a lens is given, then the value can be various length. See
   * \ref KVPairs for more information.
   *
   * The KV list is partitioned and sent based on the key range each server
   * maintaining. This function returns without waiting the data are sent
   * actually. Instead, use either \ref Wait or the callback to know when
   * finished. This function is thread-safe.
   *
   * @param keys a list of keys, must be unique and sorted in increasing order
   * @param vals the according values
   * @param lens optional, lens[i] stores the value length of the \a
   * i-th KV pair
   * @param cmd an optional command sent to the servers
   * @param last whether this is the last push in the batch
   * @param cb the callback which is called when the push is finished.
   * @return the timestamp of this request
   */
  int Push(const std::vector<Key>& keys,
           const std::vector<Val>& vals,
           const std::vector<int>& lens = {},
           int cmd = 0,
           const Callback& cb = nullptr) {
    return ZPush(
        SArray<Key>(keys), SArray<Val>(vals), SArray<int>(lens), cmd, 0, cb);
  }

  /**
   * \brief Pulls the values associated with the keys from the server nodes
   *
   * This function pulls the values of the keys specified in \a keys from the
   * server nodes. The format is same to \ref KVPairs
   *
   * Sample usage: the following codes pull the values of keys \a 1 and \a 3
   * from the server nodes.
   * \code
   *   KVWorker<float> w;
   *   std::vector<Key> keys = {1, 3};
   *   std::vector<float> vals;
   *   ps.Pull(keys, &vals);
   * \endcode
   *
   * It's a non-blocking call. The actual pulling is finished,
   * namely \a vals (and \a lens) is filled with pulled values, only
   * if \ref Wait returns or the callback is called.
   *
   * @param keys a list of keys, must be unique and sorted in increasing order
   * @param vals the buffer for the pulled values. It can be 0 size.
   * @param lens optional buffer for the value length. If set, it can be 0 size.
   * @param cmd an optional command sent to the servers
   * @param cb the callback which is called when the pull is finished.
   * @return the timestamp of this request
   */
  int Pull(const std::vector<Key>& keys,
           std::vector<Val>* vals,
           std::vector<int>* lens = nullptr,
           int cmd = 0,
           const Callback& cb = nullptr) {
    return Pull_(SArray<Key>(keys), vals, lens, cmd, cb);
  }

  /**
   * \brief Waits until a push or pull has been finished
   *
   * Sample usage:
   * \code
   *   int ts = w.Pull(keys, &vals);
   *   Wait(ts);
   *   // now vals is ready for use
   * \endcode
   *
   * \param timestamp the timestamp returned by the push or pull
   */
  void Wait(int timestamp) { obj_->WaitRequest(timestamp); }

  /**
   * \brief zero-copy Push
   *
   * This function is similar to \ref Push except that all data
   * will not be copied into system for better performance. It is the caller's
   * responsibility to keep the content to be not changed before actually
   * finished.
   */
  int ZPush(const SArray<Key>& keys,
            const SArray<Val>& vals,
            const SArray<int>& lens = {},
            int cmd = 0,
            int last = 0,
            const Callback& cb = nullptr) {
    PS_VLOG(1) << obj_ -> id() << " called ZPush";
    int ts = obj_->NewRequest(kServerGroup);
    if (MXNET_ENABLE_CANCEL && last) {
      AddCallback(ts, [this, cb]() {
        Message msg;
        msg.meta.control.cmd = Control::NOTIFY;
        msg.meta.customer_id = obj_->id();
        msg.meta.request = false;
        msg.meta.push = false;
        msg.meta.recver = Postoffice::Get()->GetNodeIDs(kScheduler).at(0);
        Postoffice::Get()->van()->Send(msg);
        // BroadcastToWorkers(msg);
        if (cb) {
          cb();
        }
      });
    } else {
      AddCallback(ts, cb);
    }
    KVPairs<Val> kvs;
    kvs.keys = keys;
    kvs.vals = vals;
    kvs.lens = lens;
    Send(ts, true, cmd, kvs);
    return ts;
  }

  /**
   * \brief zero-copy Pull
   *
   * This function is similar to \ref Pull except that all data
   * will not be copied into system for better performance. It is the caller's
   * responsibility to keep the content to be not changed before actually
   * finished.
   */
  int ZPull(const SArray<Key>& keys,
            SArray<Val>* vals,
            SArray<int>* lens = nullptr,
            int cmd = 0,
            const Callback& cb = nullptr) {
    PS_VLOG(1) << obj_ -> id() << " called ZPull";
    return Pull_(keys, vals, lens, cmd, cb);
  }
  using SlicedKVs = std::vector<std::pair<bool, KVPairs<Val>>>;
  /**
   * \brief a slicer partitions a key-value list according to the key ranges
   * \param send the kv list for partitioning
   * \param ranges the key ranges, ranges[i] is the key range of server i
   * \param sliced the sliced lists. slices[i] should only contains keys in
   * ranges[i] and the according values
   */
  using Slicer = std::function<void(
      const KVPairs<Val>& send, const std::vector<Range>& ranges,
      SlicedKVs* sliced)>;

  /**
   * \brief set a user-defined slicer
   */
  void set_slicer(const Slicer& slicer) {
    CHECK(slicer); slicer_ = slicer;
  }

  /**
   * \brief set callback to be called when cancellation is needed
   */
  void SetCancelCallback(CancelCallback callback) {
    std::lock_guard<std::mutex> lock(mu_cc);
    // LOG(INFO) << "Callback address " << (void *)callback;
    cancel_callback_ = callback;
    // cancel_callback_();
  }

 private:
  /**
   * \brief internal pull, C/D can be either SArray or std::vector
   */
  template <typename C, typename D>
  int Pull_(const SArray<Key>& keys, C* vals, D* lens,
            int cmd, const Callback& cb);
  /**
   * \brief add a callback for a request. threadsafe.
   * @param cb callback
   * @param timestamp the timestamp of the request
   */
  void AddCallback(int timestamp, const Callback& cb) {
    if (!cb) return;
    std::lock_guard<std::mutex> lk(mu_);
    callbacks_[timestamp] = cb;
  }
  /**
   * \brief broadcast the message to all workers
   */
  void BroadcastToWorkers(Message &msg);
  /**
   * \brief run and delete the callback
   * \param timestamp the timestamp of the callback
   */
  void RunCallback(int timestamp);
  /**
   * \brief send the kv list to all servers
   * @param timestamp the timestamp of the request
   * @param push whether or not it is a push request
   * @param cmd command
   */
  void Send(int timestamp, bool push, int cmd, const KVPairs<Val>& kvs);
  /** \brief internal receive handle */
  void Process(const Message& msg);
  /** \brief default kv slicer */
  void DefaultSlicer(const KVPairs<Val>& send,
                     const std::vector<Range>& ranges,
                     SlicedKVs* sliced);
  /** \brief record notification time from server */
  void RecordNotification(const int server_id);

  /** \brief data buffer for received kvs for each timestamp */
  std::unordered_map<int, std::vector<KVPairs<Val>>> recv_kvs_;
  /** \brief callbacks for each timestamp */
  std::unordered_map<int, Callback> callbacks_;
  /** \brief lock */
  std::mutex mu_;
  /** \brief kv list slicer */
  Slicer slicer_;
  /** \brief whether print profiling info */
  int MXNET_PROFILING;
  /** \brief record notification from servers */
  std::unique_ptr<NotificationMap> notification_record_;
  /** \brief mutex for notification_record_ */
  std::mutex mu_nr;
  /** \brief callback for cancellation */
  CancelCallback cancel_callback_ = nullptr;
  /** \brief mutex for cancel_callback */
  std::mutex mu_cc;
  /** \brief a flag indicating whether cancel callback is enabled */
  int MXNET_ENABLE_CANCEL;
  /** \brief window length of opportunistic restart */
  double MXNET_CANCEL_WINDOW;
  /** \brief trigger percentage of opportunistic restart */
  double MXNET_CANCEL_PERCENT;
};

/** \brief meta information about a kv request */
struct KVMeta {
  /** \brief the int cmd */
  int cmd;
  /** \brief whether or not this is a push request */
  bool push;
  /** \brief sender's node id */
  int sender;
  /** \brief the associated timestamp */
  int timestamp;
};

/**
 * \brief A server node for maintaining key-value pairs
 */
template <typename Val>
class KVServer : public SimpleApp {

 public:

  /**
   * \brief a list storing latest ten push/pull info, whose entry
   *        means (timestamp, is_push)
   */
  typedef std::list<std::pair<int, bool>> RecordList;
  /**
   * \brief a map containing workers' RecordList, whose entry means
   *        worker_id -> record_list
   */
  typedef std::map<int, RecordList> RecordMap;

  /**
   * \brief constructor
   * \param app_id the app id, should match with \ref KVWorker's id
   */
  explicit KVServer(int app_id) :
    SimpleApp(),
    time_record_(std::unique_ptr<RecordMap>(new RecordMap))
  {
    using namespace std::placeholders;
    obj_ = new Customer(app_id, std::bind(&KVServer<Val>::Process, this, _1));
    char* temp = getenv("MXNET_DELAY_SERVER");
    PULL_DELAY = temp ? atoi(temp) : 0;
  }

  /** \brief deconstructor */
  virtual ~KVServer() { delete obj_; obj_ = nullptr; }

  /**
   * \brief the handle to process a push/pull request from a worker
   * \param req_meta meta-info of this request
   * \param req_data kv pairs of this request
   * \param server this pointer
   */
  using ReqHandle = std::function<void(const KVMeta& req_meta,
                                       const KVPairs<Val>& req_data,
                                       KVServer* server)>;
  void set_request_handle(const ReqHandle& request_handle) {
    CHECK(request_handle) << "invalid request handle";
    request_handle_ = request_handle;
  }

  /**
   * \brief response to the push/pull request
   * \param req the meta-info of the request
   * \param res the kv pairs that will send back to the worker
   */
  void Response(const KVMeta& req, const KVPairs<Val>& res = KVPairs<Val>());

  /**
   * \brief broadcast message to all workers
   */
  void Broadcast(Message &msg);

  /**
   * \brief get the id of customer
   */
  int id() { return obj_->id(); }

 private:
  /** \brief internal receive handle */
  void Process(const Message& msg);
  /** \brief request handle */
  ReqHandle request_handle_;
  /** \brief record of workers pulling and pushing */
  std::unique_ptr<RecordMap> time_record_;
  /** \brief period length of pulling delay  */
  int PULL_DELAY;
};


/**
 * \brief an example handle adding pushed kv into store
 */
template <typename Val>
struct KVServerDefaultHandle {
  void operator()(
      const KVMeta& req_meta, const KVPairs<Val>& req_data, KVServer<Val>* server) {
    size_t n = req_data.keys.size();
    KVPairs<Val> res;
    for (size_t i = 0; i < n; ++i) {
      Key key = req_data.keys[i];
      if (req_meta.push) {
        store[key] += req_data.vals[i];
      } else {
        res.vals[i] = store[key];
      }
    }
    server->Response(req_meta, res);
  }
  std::unordered_map<Key, Val> store;
};


///////////////////////////////////////////////////////////////////////////////

template<typename Val>
void KVServer<Val>::Broadcast(Message &msg) {
  for (int id : Postoffice::Get()->GetNodeIDs(kWorkerGroup)) {
    msg.meta.recver = id;
    Postoffice::Get()->van()->Send(msg);
  }
}

template <typename Val>
void KVServer<Val>::Process(const Message& msg) {
  if (msg.meta.simple_app) {
    SimpleApp::Process(msg); return;
  }
  KVMeta meta;
  meta.cmd       = msg.meta.head;
  meta.push      = msg.meta.push;
  meta.sender    = msg.meta.sender;
  meta.timestamp = msg.meta.timestamp;

  KVPairs<Val> data;
  int n = msg.data.size();
  if (n) {
    CHECK_GE(n, 2);
    data.keys = msg.data[0];
    data.vals = msg.data[1];
    if (n > 2) {
      CHECK_EQ(n, 3);
      data.lens = msg.data[2];
      CHECK_EQ(data.lens.size(), data.keys.size());
    }
  }

  /*
  if (time_record_->find(meta.sender) == time_record_->end()) {
    time_record_->insert({meta.sender, { {meta.timestamp, meta.push} }});
  } else {
    auto &record_list = time_record_->at(meta.sender);
    record_list.emplace_back(meta.timestamp, meta.push);
    if (record_list.size() > 10) {
      record_list.pop_front();
    }
  }
  */

  /*
  std::stringstream ss;
  for (auto &p : *time_record_) {
    ss << p.first << " : ";
    for (auto &r : p.second) {
      ss << "(" << r.first << "," << r.second << ") ";
    }
  }
  LOG(INFO) << ss.str();
  */

  CHECK(request_handle_);
  request_handle_(meta, data, this);
}

template <typename Val>
void KVServer<Val>::Response(const KVMeta& req, const KVPairs<Val>& res) {
  Message msg;
  msg.meta.customer_id = obj_->id();
  msg.meta.request     = false;
  msg.meta.push        = req.push;
  msg.meta.head        = req.cmd;
  msg.meta.timestamp   = req.timestamp;
  msg.meta.recver      = req.sender;
  if (res.keys.size()) {
    msg.AddData(res.keys);
    msg.AddData(res.vals);
    if (res.lens.size()) {
      msg.AddData(res.lens);
    }
  }
  Postoffice::Get()->van()->Send(msg);
}

template <typename Val>
void KVWorker<Val>::BroadcastToWorkers(Message &msg) {
  for (int id : Postoffice::Get()->GetNodeIDs(kWorkerGroup)) {
    msg.meta.recver = id;
    Postoffice::Get()->van()->Send(msg);
  }
}

template <typename Val>
void KVWorker<Val>::DefaultSlicer(
    const KVPairs<Val>& send, const std::vector<Range>& ranges,
    typename KVWorker<Val>::SlicedKVs* sliced) {
  sliced->resize(ranges.size());

  // find the positions in msg.key
  size_t n = ranges.size();
  std::vector<size_t> pos(n+1);
  const Key* begin = send.keys.begin();
  const Key* end = send.keys.end();
  for (size_t i = 0; i < n; ++i) {
    if (i == 0) {
      pos[0] = std::lower_bound(begin, end, ranges[0].begin()) - begin;
      begin += pos[0];
    } else {
      CHECK_EQ(ranges[i-1].end(), ranges[i].begin());
    }
    size_t len = std::lower_bound(begin, end, ranges[i].end()) - begin;
    begin += len;
    pos[i+1] = pos[i] + len;

    // don't send it to severs for empty kv
    sliced->at(i).first = (len != 0);
  }
  CHECK_EQ(pos[n], send.keys.size());
  if (send.keys.empty()) return;

  // the length of value
  size_t k = 0, val_begin = 0, val_end = 0;
  if (send.lens.empty()) {
    k = send.vals.size() / send.keys.size();
    CHECK_EQ(k * send.keys.size(), send.vals.size());
  } else {
    CHECK_EQ(send.keys.size(), send.lens.size());
  }

  // slice
  for (size_t i = 0; i < n; ++i) {
    if (pos[i+1] == pos[i]) {
      sliced->at(i).first = false;
      continue;
    }
    sliced->at(i).first = true;
    auto& kv = sliced->at(i).second;
    kv.keys = send.keys.segment(pos[i], pos[i+1]);
    if (send.lens.size()) {
      kv.lens = send.lens.segment(pos[i], pos[i+1]);
      for (int l : kv.lens) val_end += l;
      kv.vals = send.vals.segment(val_begin, val_end);
      val_begin = val_end;
    } else {
      kv.vals = send.vals.segment(pos[i]*k, pos[i+1]*k);
    }
  }
}

template <typename Val>
void KVWorker<Val>::Send(int timestamp, bool push, int cmd, const KVPairs<Val>& kvs) {
  // slice the message
  SlicedKVs sliced;
  slicer_(kvs, Postoffice::Get()->GetServerKeyRanges(), &sliced);

  // need to add response first, since it will not always trigger the callback
  int skipped = 0;
  for (size_t i = 0; i < sliced.size(); ++i) {
    if (!sliced[i].first) ++skipped;
  }
  obj_->AddResponse(timestamp, skipped);
  if ((size_t)skipped == sliced.size()) {
    RunCallback(timestamp);
  }

  for (size_t i = 0; i < sliced.size(); ++i) {
    const auto& s = sliced[i];
    if (!s.first) continue;
    Message msg;
    msg.meta.customer_id = obj_->id();
    msg.meta.request     = true;
    msg.meta.push        = push;
    msg.meta.head        = cmd;
    msg.meta.timestamp   = timestamp;
    msg.meta.recver      = Postoffice::Get()->ServerRankToID(i);

    if (MXNET_PROFILING) {
      /*
      time_t current;
      time(&current);
      LOG(INFO) << Postoffice::Get()->my_rank()
                << (msg.meta.push ? ">" : "<")
                << msg.meta.recver
                << "," << msg.meta.timestamp
                << "," << current << ",";
      */
    }

    const auto& kvs = s.second;
    if (kvs.keys.size()) {
      msg.AddData(kvs.keys);
      msg.AddData(kvs.vals);
      if (kvs.lens.size()) {
        msg.AddData(kvs.lens);
      }
    }
    Postoffice::Get()->van()->Send(msg);
  }
}

template <typename Val>
void KVWorker<Val>::RecordNotification(const int worker_id) {
  std::lock_guard<std::mutex> lock(mu_nr);

  if (!cancel_callback_) {
    return;
  }

  using namespace std::chrono;
  auto now = system_clock::now().time_since_epoch();
  double timestamp = duration_cast<milliseconds>(now).count() / 1000.0;

  if (notification_record_->find(worker_id) == notification_record_->end()) {
    notification_record_->insert({worker_id, timestamp});
  } else {
    notification_record_->at(worker_id) = timestamp;
    double updated_within_window = 0;
    double threshold = notification_record_->size() * MXNET_CANCEL_PERCENT;
    for (auto &p : *notification_record_) {
      if (timestamp - p.second <= MXNET_CANCEL_WINDOW) {
        updated_within_window += 1;
        if (updated_within_window > threshold) {
          cancel_callback_();
          break;
        }
      }
    }
  }

  /*
  std::stringstream ss;
  ss << "{ ";
  for (auto &p : *notification_record_) {
    ss  << "{" << p.first << ", " << p.second << "} ";
  }
  ss << "}";
  LOG(INFO) << ss.str();
  */
}

template <typename Val>
void KVWorker<Val>::Process(const Message& msg) {

  // LOG(INFO) << "Received " << msg.DebugString();

  if (msg.meta.simple_app) {
    SimpleApp::Process(msg);
    return;
  }
  
  if (MXNET_ENABLE_CANCEL &&
      msg.meta.control.cmd == Control::CANCEL) {
    std::lock_guard<std::mutex> lock(mu_cc);
    // LOG(INFO) << "Before calling cancel_callback_";
    if (cancel_callback_) {
        cancel_callback_();
    } else {
        LOG(INFO) << "cancel_callback_ is NULL.";
    }
    /*
    if (cancel_callback_) {
      new std::thread([this] () { 
        LOG(INFO) << "attempt to call cancel_callback_";
        this->cancel_callback_(); 
      });
    }
    */
    return;
  }


  // store the data for pulling
  int ts = msg.meta.timestamp;
  if (!msg.meta.push && msg.data.size()) {
    CHECK_GE(msg.data.size(), (size_t)2);
    KVPairs<Val> kvs;
    kvs.keys = msg.data[0];
    kvs.vals = msg.data[1];
    if (msg.data.size() > (size_t)2) {
      kvs.lens = msg.data[2];
    }
    mu_.lock();
    recv_kvs_[ts].push_back(kvs);
    mu_.unlock();
  }

  // finished, run callbacks
  if (obj_->NumResponse(ts) == Postoffice::Get()->num_servers() - 1)  {
    RunCallback(ts);
  }
}

template <typename Val>
void KVWorker<Val>::RunCallback(int timestamp) {
  mu_.lock();
  auto it = callbacks_.find(timestamp);
  if (it != callbacks_.end()) {
    mu_.unlock();

    CHECK(it->second);
    it->second();

    mu_.lock();
    callbacks_.erase(it);
  }
  mu_.unlock();
}

template <typename Val>
template <typename C, typename D>
int KVWorker<Val>::Pull_(
    const SArray<Key>& keys, C* vals, D* lens, int cmd, const Callback& cb) {
  int ts = obj_->NewRequest(kServerGroup);
  AddCallback(ts, [this, ts, keys, vals, lens, cb]() mutable {
      mu_.lock();
      auto& kvs = recv_kvs_[ts];
      mu_.unlock();

      // do check
      size_t total_key = 0, total_val = 0;
      for (const auto& s : kvs) {
        Range range = FindRange(keys, s.keys.front(), s.keys.back()+1);
        CHECK_EQ(range.size(), s.keys.size())
            << "unmatched keys size from one server";
        if (lens) CHECK_EQ(s.lens.size(), s.keys.size());
        total_key += s.keys.size();
        total_val += s.vals.size();
      }
      CHECK_EQ(total_key, keys.size()) << "lost some servers?";

      // fill vals and lens
      std::sort(kvs.begin(), kvs.end(), [](
          const KVPairs<Val>& a, const KVPairs<Val>& b) {
                  return a.keys.front() < b.keys.front();
        });
      CHECK_NOTNULL(vals);
      if (vals->empty()) {
        vals->resize(total_val);
      } else {
        CHECK_EQ(vals->size(), total_val);
      }
      Val* p_vals = vals->data();
      int *p_lens = nullptr;
      if (lens) {
        if (lens->empty()) {
          lens->resize(keys.size());
        } else {
          CHECK_EQ(lens->size(), keys.size());
        }
        p_lens = lens->data();
      }
      for (const auto& s : kvs) {
        memcpy(p_vals, s.vals.data(), s.vals.size() * sizeof(Val));
        p_vals += s.vals.size();
        if (p_lens) {
          memcpy(p_lens, s.lens.data(), s.lens.size() * sizeof(int));
          p_lens += s.lens.size();
        }
      }

      mu_.lock();
      recv_kvs_.erase(ts);
      mu_.unlock();
      if (cb) cb();
    });

  KVPairs<Val> kvs; kvs.keys = keys;
  Send(ts, false, cmd, kvs);
  return ts;
}

}  // namespace ps
#endif  // PS_KV_APP_H_
