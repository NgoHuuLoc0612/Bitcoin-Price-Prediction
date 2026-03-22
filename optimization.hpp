/**
 * optimization.hpp — Optimizer declarations, schedulers, thread pool
 */
#pragma once
#include "core.hpp"
#include <vector>
#include <deque>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

// ─── NUMA Allocator ──────────────────────────────────────────
class NUMAAllocator {
public:
    static NUMAAllocator& instance();
    void* alloc(size_t bytes, int node = -1);
    void  free(void* ptr, size_t bytes);
    std::vector<float> alloc_vector(size_t n, int node = -1);
    int num_nodes() const { return num_nodes_; }
    bool numa_available() const { return numa_available_; }
private:
    NUMAAllocator();
    bool numa_available_;
    int  num_nodes_;
};

// ─── Optimizers ───────────────────────────────────────────────
struct AdamOptimizer {
    float lr, beta1, beta2, eps, weight_decay;
    int t;
    std::vector<float> m, v;
    AdamOptimizer(float lr=1e-3f, float beta1=0.9f, float beta2=0.999f,
                  float eps=1e-8f, float wd=0.0f);
    void init(const std::vector<float>& params);
    void step(std::vector<float>& params, const std::vector<float>& grads);
    void reset();
};

struct RMSPropOptimizer {
    float lr, alpha, eps;
    std::vector<float> v;
    RMSPropOptimizer(float lr=1e-3f, float alpha=0.99f, float eps=1e-8f);
    void step(std::vector<float>& params, const std::vector<float>& grads);
};

struct AdaGradOptimizer {
    float lr, eps;
    std::vector<float> G;
    AdaGradOptimizer(float lr=1e-2f, float eps=1e-8f);
    void step(std::vector<float>& params, const std::vector<float>& grads);
};

struct LBFGSOptimizer {
    float lr;
    int m;
    std::deque<std::vector<float>> s_list, y_list;
    LBFGSOptimizer(float lr=1.0f, int m=10);
    void step(std::vector<float>& params, const std::vector<float>& grads);
private:
    float dot(const std::vector<float>& a, const std::vector<float>& b);
    void  axpy(float alpha, const std::vector<float>& x, std::vector<float>& y);
};

// ─── Gradient Clipping ────────────────────────────────────────
struct GradientClipper {
    static void clip_by_norm (std::vector<float>& grads, float max_norm);
    static void clip_by_value(std::vector<float>& grads, float min_v, float max_v);
};

// ─── LR Scheduler ─────────────────────────────────────────────
struct LRScheduler {
    enum class SchedulerType { CONSTANT, COSINE_ANNEALING, CYCLICAL, ONE_CYCLE, REDUCE_ON_PLATEAU };

    float base_lr, max_lr = 0.01f, current_lr;
    SchedulerType type;
    int warmup_steps, step_;
    float best_loss;
    int patience_counter;

    LRScheduler(float base_lr, SchedulerType type = SchedulerType::COSINE_ANNEALING,
                int warmup_steps = 50);
    float get_lr();
    void  step_on_loss(float loss);
};

// ─── ThreadPool ────────────────────────────────────────────────
class ThreadPool {
public:
    explicit ThreadPool(int n = std::thread::hardware_concurrency());
    ~ThreadPool();
    void enqueue(std::function<void()> task);
    void wait_all();
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

// ─── Train Config / Result ────────────────────────────────────
struct TrainConfig {
    int   epochs                = 100;
    float lr                    = 1e-3f;
    float weight_decay          = 1e-5f;
    int   batch_size            = 32;
    int   patience              = 10;
    bool  use_gradient_clipping = true;
    float clip_norm             = 1.0f;     // alias used by bindings
    float gradient_clip         = 1.0f;     // kept for internal use
    float early_stop_threshold  = 1e-4f;
};

struct TrainResult {
    // Summary fields expected by bindings.cpp
    float final_loss  = 0.0f;
    float best_loss   = 1e9f;
    int   epochs_run  = 0;
    bool  converged   = false;
    std::vector<float> loss_history;

    // Per-epoch detail (used internally by BatchTrainer)
    int   epoch       = 0;
    float train_loss  = 0.0f;
    float val_loss    = 0.0f;
    float lr          = 0.0f;
    long long start_time = 0, end_time = 0;
    float throughput  = 0.0f; // samples/sec
};

// ─── BatchTrainer ──────────────────────────────────────────────
class BatchTrainer {
public:
    BatchTrainer(EnsemblePredictor& model, const TrainConfig& cfg);
    TrainResult train_epoch(const std::vector<std::vector<float>>& X,
                             const std::vector<float>& y);
    std::vector<TrainResult> train(const std::vector<std::vector<float>>& X,
                                    const std::vector<float>& y, int epochs);
    const std::vector<TrainResult>& history() const { return train_history; }
private:
    EnsemblePredictor& model;
    TrainConfig cfg;
    AdamOptimizer optimizer;
    LRScheduler scheduler;
    ThreadPool pool;
    std::vector<TrainResult> train_history;
    int epoch_ = 0;
};

// ─── Backtest ──────────────────────────────────────────────────
struct BacktestResult {
    float initial_capital;
    float final_capital;
    float total_return;
    float max_drawdown   = 0.0f;
    float sharpe_ratio   = 0.0f;
    float win_rate       = 0.0f;
    int   num_trades     = 0;
    std::vector<float> equity_curve;
};

struct BacktestEngine {
    // Signature matching bindings.cpp: run(prices, predictor, capital, cost)
    static BacktestResult run(const std::vector<float>& prices,
                               RealTimePredictor& predictor,
                               float initial_capital  = 10000.0f,
                               float transaction_cost = 0.001f);
};

// ─── Hyperparameter Search ────────────────────────────────────
struct AutoHyperparamSearch {
    int n_trials = 20;
    explicit AutoHyperparamSearch(int n_trials = 20) : n_trials(n_trials) {}

    // Instance method expected by bindings.cpp
    ModelConfig search(const std::vector<float>& prices, int n = 0) const {
        return random_search(prices, n > 0 ? n : n_trials);
    }

    // Original static entry point (kept for internal use)
    static ModelConfig random_search(const std::vector<float>& prices, int n_trials = 20);
};
