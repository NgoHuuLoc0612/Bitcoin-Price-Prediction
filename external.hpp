/**
 * external.hpp — External data connectors declarations
 */
#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>
#include <map>
#include <functional>
#include <memory>

// ─── OHLCV Candle ────────────────────────────────────────────
struct OHLCV {
    long long timestamp = 0;
    float open = 0, high = 0, low = 0, close = 0, volume = 0;
    std::vector<float> to_vector() const;
};

// ─── Thread-safe price buffer ────────────────────────────────
class PriceBuffer {
public:
    explicit PriceBuffer(size_t max_size = 2000) : max_size_(max_size) {}
    void push(const OHLCV& candle);
    std::vector<OHLCV>  snapshot() const;
    std::vector<float>  close_prices() const;
    size_t size() const;
    uint64_t version() const;
private:
    mutable std::mutex mtx;
    std::vector<OHLCV> buffer_;
    size_t max_size_;
    std::atomic<uint64_t> version_{0};
};

// ─── CoinGecko ───────────────────────────────────────────────
struct CoinGeckoMarketData {
    float price           = 0;
    float market_cap      = 0;
    float volume_24h      = 0;
    float price_change_24h= 0;
    float circulating_supply = 0;
    float total_supply    = 0;
    float ath             = 0;
};

class CoinGeckoClient {
public:
    explicit CoinGeckoClient(const std::string& api_key = "");
    std::vector<OHLCV>    fetch_ohlcv(const std::string& coin_id = "bitcoin",
                                       const std::string& vs = "usd", int days = 365);
    float                 fetch_current_price(const std::string& coin_id = "bitcoin",
                                               const std::string& vs = "usd");
    CoinGeckoMarketData   fetch_market_data(const std::string& coin_id = "bitcoin");
    const std::string&    last_error() const { return last_error_; }
private:
    std::string api_key_, base_url_, last_error_;
    long long last_request_;
    void rate_limit_wait();
};

// ─── Binance WebSocket (polling fallback) ────────────────────
class BinanceWSClient {
public:
    BinanceWSClient(PriceBuffer& buffer, const std::string& symbol = "BTCUSDT");
    ~BinanceWSClient();
    void start();
    void stop();
    void set_on_price(std::function<void(float)> cb) { on_price_ = cb; }
    int  error_count() const { return error_count_.load(); }
    void set_poll_interval(int ms) { poll_interval_ms_ = ms; }
private:
    PriceBuffer& buffer_;
    std::string symbol_;
    std::atomic<bool> running_;
    std::thread worker_;
    std::function<void(float)> on_price_;
    std::atomic<int> error_count_{0};
    int poll_interval_ms_ = 5000;
    void poll_loop();
};

// ─── Order Book ──────────────────────────────────────────────
class OrderBookAggregator {
public:
    void  update_bid(float price, float qty);
    void  update_ask(float price, float qty);
    float best_bid() const;
    float best_ask() const;
    float mid_price() const;
    float spread() const;
    float bid_ask_imbalance() const;
private:
    mutable std::mutex mtx_;
    std::map<float, float> bids_;   // price → qty, descending
    std::map<float, float> asks_;   // price → qty, ascending
};

// ─── Sentiment ───────────────────────────────────────────────
struct SentimentScore {
    float fear_greed_index = 50.0f;
    float bullish_score    = 0.5f;
    float bearish_score    = 0.5f;
    std::string classification = "neutral";
};

struct SentimentAnalyzer {
    static SentimentScore fetch_fear_greed();
};

// ─── On-chain metrics ────────────────────────────────────────
struct OnChainMetrics {
    long  active_addresses  = 0;
    long  transaction_count = 0;
    float hash_rate_th_s    = 0;
    float difficulty        = 0;
    int   mempool_size      = 0;
    float exchange_inflows  = 0;
    float exchange_outflows = 0;
    float nvt_ratio         = 0;
    float sopr              = 1.0f;
};

struct OnChainDataProvider {
    static OnChainMetrics fetch_metrics();
};

// ─── Pipeline stats ──────────────────────────────────────────
struct PipelineStats {
    int buffer_size;
    uint64_t data_version;
    int ws_errors;
    long long last_update_ms;
};

// ─── DataPipeline ────────────────────────────────────────────
class DataPipeline {
public:
    explicit DataPipeline(const std::string& coingecko_key = "");
    ~DataPipeline();
    bool start();
    void stop();
    PriceBuffer& buffer() { return price_buffer_; }
    PipelineStats stats() const;
    void set_on_new_price(std::function<void(float)> cb) { on_new_price_ = cb; }
private:
    CoinGeckoClient coingecko_;
    PriceBuffer price_buffer_;
    std::unique_ptr<BinanceWSClient> binance_ws_;
    std::atomic<bool> running_;
    std::thread refresh_thread_;
    std::function<void(float)> on_new_price_;
};
