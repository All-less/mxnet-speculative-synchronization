/*!
 * Copyright (c) 2015 by Contributors
 * \file threaded_engine_pooled.cc
 * \brief Pooled threaded engine
 * \author Yutian Li
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/concurrency.h>
#include <cassert>
#include "./threaded_engine.h"
#include "./thread_pool.h"
#include "./stream_manager.h"

namespace mxnet {
namespace engine {
/*!
 * \brief ThreadedEngine using global thread pool across all devices.
 * The policy of this Engine:
 *  - Execute Async operation immediately if pushed from Pusher.
 *  - Use a common thread pool for normal operations on all devices.
 *  - Use special thread pool for copy operations.
 */
class ThreadedEnginePooled : public ThreadedEngine {
 public:
  ThreadedEnginePooled() :
      thread_pool_(kNumWorkingThreads, [this](int index) { IndexedThreadWorker(&task_queue_, index); }),
      io_thread_pool_(1, [this, kNumWorkingThreads](int index) { IndexedThreadWorker(&io_task_queue_, kNumWorkingThreads); }) {}

  ~ThreadedEnginePooled() noexcept(false) {
    streams_.Finalize();
    task_queue_.SignalForKill();
    io_task_queue_.SignalForKill();
  }

  void ClearTask() override {
    // LOG(INFO) << "ClearTask() begins";
    stop_flag_.store(true);
    for (int i = 0; i < kNumWorkingThreads; i++) {
      /*!
       * Several dummy operators are pushed to ensure that all workers
       * are waiting on cv_continue_.
       */
      task_queue_.Push(new OprBlock());
    }
    io_task_queue_.Push(new OprBlock());
    for (auto &b : stopped_) {
      /*!
       * We wait for all workers to be waiting on cv_continue_.
       */
      while (!b.load()) { }
    }
    {
      std::lock_guard<std::mutex> lock(mu_continue_);
      OprBlock* opr_block;
      while (task_queue_.Size() > 0) {
        task_queue_.Pop(&opr_block);
        opr_block->ClearDependencies();
      }
      stop_flag_.store(false);
    }
    cv_continue_.notify_all();
    for (auto &b : cleared_) {
      while (!b.load()) { }
    }
    {
      std::lock_guard<std::mutex> lock(mu_continue_);
    }
    cv_continue_.notify_all();
    // LOG(INFO) << "ClearTask() ends";
  }

 protected:
  void PushToExecute(OprBlock *opr_block, bool pusher_thread) override {
    if (opr_block->opr->prop == FnProperty::kAsync && pusher_thread) {
      DoExecute(opr_block);
    } else {
      DoPushToQueue(opr_block);
    }
  }

 private:
  /*! \brief Concurrency for thread pool */
  static constexpr std::size_t kNumWorkingThreads = 16;
  /*! \brief Maximum number of GPUs */
  static constexpr std::size_t kMaxNumGpus = 16;
  /*!\brief number of streams allocated for each GPU */
  static constexpr std::size_t kNumStreamsPerGpu = 16;
  /*!
   * \brief Streams.
   */
  StreamManager<kMaxNumGpus, kNumStreamsPerGpu> streams_;
  /*!
   * \brief Task queues.
   */
  dmlc::ConcurrentBlockingQueue<OprBlock*> task_queue_;
  dmlc::ConcurrentBlockingQueue<OprBlock*> io_task_queue_;
  /*!
   * \brief Thread pools.
   */
  ThreadPool thread_pool_;
  ThreadPool io_thread_pool_;
  /*!
   * \brief a flag informing workers that we want to stop now
   */
  std::atomic<bool> stop_flag_ {false};
  /*!
   * \brief condition variable via which workers know when to restart work
   */
  std::mutex mu_continue_;
  std::condition_variable cv_continue_;
  /*!
   * \brief flags indicating whether workers have stopped
   */
  std::atomic<bool> stopped_[kNumWorkingThreads + 1];
  /*!
   * \brief flags indicating whether workers have cleared their dependencies
   */
  std::atomic<bool> cleared_[kNumWorkingThreads + 1];
  /*!
   * \brief Worker.
   * \param task_queue Queue to work on.
   *
   * The method to pass to thread pool to parallelize.
   */
  void ThreadWorker(dmlc::ConcurrentBlockingQueue<OprBlock*>* task_queue) {
    OprBlock* opr_block;
    while (task_queue->Pop(&opr_block)) {
      DoExecute(opr_block);
    }
  }
  /*!
   * \brief worker with index as an argument
   */
  void IndexedThreadWorker(dmlc::ConcurrentBlockingQueue<OprBlock*>* task_queue, int index) {
    stopped_[index].store(false);
    cleared_[index].store(false);
    OprBlock* opr_block;
    while (task_queue->Pop(&opr_block)) {
      if (stop_flag_.load()) {
        {
          std::unique_lock<std::mutex> lock(mu_continue_);
          stopped_[index].store(true);
          cv_continue_.wait(lock);
          stopped_[index].store(false);
        }
        {
          std::unique_lock<std::mutex> lock(mu_continue_);
          opr_block->ClearDependencies();
          cleared_[index].store(true);
          cv_continue_.wait(lock);
          cleared_[index].store(false);
        }
        continue;
      }
      /*
      if (opr_block && opr_block->opr && opr_block->opr->opr_name) {
        LOG(INFO) << task_queue->Size()
                  << " Execute : " <<  opr_block->opr->opr_name
                  << " " << opr_block->opr->const_vars.size()
                  << " " << opr_block->opr->mutable_vars.size();
      }
      */
      DoExecute(opr_block);
    }
  }
  /*!
   * \brief Execute an operation.
   * \param opr_block The operator block.
   */
  void DoExecute(OprBlock* opr_block) {
    assert(opr_block->wait.load() == 0);
    if (opr_block->ctx.dev_mask() == gpu::kDevMask) {
      #if MXNET_USE_CUDA
      CUDA_CALL(cudaSetDevice(opr_block->ctx.dev_id));
      #else   // MXNET_USE_CUDA
      LOG(FATAL) << "Please compile with CUDA enabled";
      #endif  // MXNET_USE_CUDA
    }
    bool is_copy = (opr_block->opr->prop == FnProperty::kCopyFromGPU ||
                    opr_block->opr->prop == FnProperty::kCopyToGPU);
    auto&& rctx = is_copy
        ? streams_.GetIORunContext(opr_block->ctx)
        : streams_.GetRunContext(opr_block->ctx);
    this->ExecuteOprBlock(rctx, opr_block);
  }
  /*!
   * \brief Push the operation to the queue.
   * \param opr_block The operator block.
   */
  void DoPushToQueue(OprBlock* opr_block) {
    /*
    LOG(INFO) << task_queue_.Size()
              << " Push : " <<  opr_block->opr->opr_name
              << " " << opr_block->opr->const_vars.size()
              << " " << opr_block->opr->mutable_vars.size();
    */
    switch (opr_block->opr->prop) {
      case FnProperty::kCopyFromGPU:
      case FnProperty::kCopyToGPU: {
        io_task_queue_.Push(opr_block);
        break;
      }
      default: {
        task_queue_.Push(opr_block);
        break;
      }
    }
  }
};

Engine *CreateThreadedEnginePooled() {
  return new ThreadedEnginePooled();
}
}  // namespace engine
}  // namespace mxnet
