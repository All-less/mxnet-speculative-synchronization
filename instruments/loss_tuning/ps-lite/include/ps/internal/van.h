/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_INTERNAL_VAN_H_
#define PS_INTERNAL_VAN_H_
#include <unordered_map>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <memory>
#include <atomic>
#include <ctime>
#include <set>
#include <mutex>
#include <queue>
#include <list>
#include "ps/base.h"
#include "ps/internal/message.h"
namespace ps {
class Resender;
/**
 * \brief Van sends messages to remote nodes
 *
 * If environment variable PS_RESEND is set to be 1, then van will resend a
 * message if it no ACK messsage is received within PS_RESEND_TIMEOUT millisecond
 */
class Van {
 public:
  /**
   * \brief create Van
   * \param type zmq, socket, ...
   */
  static Van* Create(const std::string& type);
  /** \brief constructer, do nothing. use \ref Start for real start */
  Van() { }
  /**\brief deconstructer, do nothing. use \ref Stop for real stop */
  virtual ~Van() { }
  /**
   * \brief start van
   *
   * must call it before calling Send
   *
   * it initalizes all connections to other nodes.  start the receiving
   * threads, which keeps receiving messages. if it is a system
   * control message, give it to postoffice::manager, otherwise, give it to the
   * accoding app.
   */
  virtual void Start();
  /**
   * \brief send a message, It is thread-safe
   * \return the number of bytes sent. -1 if failed
   */
  int Send(const Message& msg);
  /**
   * \brief return my node
   */
  const Node& my_node() const {
    CHECK(ready_) << "call Start() first";
    return my_node_;
  }
  /**
   * \brief stop van
   * stop receiving threads
   */
  virtual void Stop();
  /**
   * \brief get next available timestamp. thread safe
   */
  int GetTimestamp() { return timestamp_++; }
  /**
   * \brief whether it is ready for sending. thread safe
   */
  bool IsReady() { return ready_; }
  /**
   * \brief several typing shorthands
   */
  typedef std::list<std::chrono::milliseconds> LengthList;
  typedef std::vector<std::chrono::system_clock::time_point> TimeVector;
  typedef std::pair<double, std::time_t> LossPair;
  typedef std::vector<LossPair> LossVector;
  typedef struct {
    /** worker id */
    int id;
    /** predicted time */
    std::chrono::system_clock::time_point time;
    /** variance of prediction */
    double var;
  } PredTime;
  /**
   * PREDICT : predict optimal wait_time
   * CHECK : check if the condition of restart is met
   */
  enum Task { PREDICT, CHECK };
  typedef struct TaskSpec {
    /** execute 'task' for worker 'id' at 'time' */
    Task task;
    int id;
    std::chrono::system_clock::time_point time;
  public:
    bool operator< (const TaskSpec t) const {
      /**
       * TaskSpec1 < TaskSpec2 means TaskSpec1 will
       * be executed later than TaskSpec2.
       */
      return time > t.time;
    }
  } TaskSpec;

  enum CancelState {
    SKIP_FIRST, /* skip first epoch              */
    TRYING,     /* trying different wait_time    */
    SETTLED     /* decided wait_tiem             */
  };

 protected:
  /**
   * \brief connect to a node
   */
  virtual void Connect(const Node& node) = 0;
  /**
   * \brief bind to my node
   * do multiple retries on binding the port. since it's possible that
   * different nodes on the same machine picked the same port
   * \return return the port binded, -1 if failed.
   */
  virtual int Bind(const Node& node, int max_retry) = 0;
  /**
   * \brief block until received a message
   * \return the number of bytes received. -1 if failed or timeout
   */
  virtual int RecvMsg(Message* msg) = 0;
  /**
   * \brief send a mesage
   * \return the number of bytes sent
   */
  virtual int SendMsg(const Message& msg) = 0;
  /**
   * \brief pack meta into a string
   */
  void PackMeta(const Meta& meta, char** meta_buf, int* buf_size);
  /**
   * \brief unpack meta from a string
   */
  void UnpackMeta(const char* meta_buf, int buf_size, Meta* meta);

  Node scheduler_;
  Node my_node_;
  bool is_scheduler_;

 private:
  /** thread function for receving */
  void Receiving();
  /** thread function for heartbeat */
  void Heartbeat();
  /** initialize all cancellation-related data structure */
  void InitializeCancellation();
  /** execute CHECK and PREDICT tasks */
  void ExecuteTasks();
  /** check whether the condition of cancellation is met */
  void CheckCancellation(int id, std::chrono::system_clock::time_point &now);
  /** get trial wait_time */
  std::chrono::milliseconds GetTrialWaitTime(int epoch_num);
  /** update cancellation-related information */
  void OnRecvNotif(Message &msg);
  /** update progress ratio */
  void OnRecvReport(Message &msg);
  /** update cancellation threshold */
  void UpdateThreshold();
  /** find optimal wait time based on loss trace */
  void SetOptimalWaitTime();

  /** whether it is ready for sending */
  std::atomic<bool> ready_{false};
  std::atomic<size_t> send_bytes_{0};
  size_t recv_bytes_ = 0;
  int num_servers_ = 0;
  int num_workers_ = 0;
  /** the thread for receiving messages */
  std::unique_ptr<std::thread> receiver_thread_;
  /** the thread for sending heartbeat */
  std::unique_ptr<std::thread> heartbeat_thread_;
  /** the thread for sending cancelling */
  std::unique_ptr<std::thread> cancel_thread_;
  /** loss and time of each epoch of each worker */
  std::unique_ptr<std::vector<LossVector>> loss_trace_;
  /** index of current push */
  int push_num_;
  /** timestamps of each push */
  std::unique_ptr<TimeVector> push_trace_;
  /** number of epochs */
  int epoch_num_;
  /** length of previous epoch */
  std::chrono::milliseconds epoch_length_;
  /** wait_time for current epoch */
  std::chrono::milliseconds wait_time_;
  /** a list of workers waiting for possible cancellation */
  std::unique_ptr<std::set<int>> waiting_workers_;
  /** number of fresh updates each worker */
  std::unique_ptr<std::vector<int>> update_count_;
  /** mutex for visiting cancellation-related data */
  std::mutex cancel_mutex_;
  /** tasks to be executed in the future, in ascending order of time */
  std::unique_ptr<std::priority_queue<TaskSpec>> task_queue_;
  /** number of fresh updates to restart computation */
  double cancel_threshold_;
  /** upper bound of ratio between wait_time and epoch length */
  double upper_ratio_;

  std::vector<int> barrier_count_;
  /** msg resender */
  Resender* resender_ = nullptr;
  int drop_rate_ = 0;
  std::atomic<int> timestamp_{0};
  DISALLOW_COPY_AND_ASSIGN(Van);
};
}  // namespace ps
#endif  // PS_INTERNAL_VAN_H_
