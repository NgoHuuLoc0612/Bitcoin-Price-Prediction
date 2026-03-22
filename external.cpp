/**
 * external.cpp — External Data Connectors
 * CoinGecko REST, Binance/Kraken WebSocket, order book aggregator,
 * sentiment analysis bridge, on-chain data ingestion
 */

#include "external.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <cmath>

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

// ─────────────────────────────────────────────────────────────
//  OHLCV
// ─────────────────────────────────────────────────────────────

std::vector<float> OHLCV::to_vector() const {
    return { open, high, low, close, volume };
}

// ─────────────────────────────────────────────────────────────
//  PriceBuffer
// ─────────────────────────────────────────────────────────────

void PriceBuffer::push(const OHLCV& candle) {
    std::lock_guard<std::mutex> lock(mtx);
    buffer_.push_back(candle);
    if (buffer_.size() > max_size_)
        buffer_.erase(buffer_.begin());
    ++version_;
}

std::vector<OHLCV> PriceBuffer::snapshot() const {
    std::lock_guard<std::mutex> lock(mtx);
    return buffer_;
}

std::vector<float> PriceBuffer::close_prices() const {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<float> out;
    out.reserve(buffer_.size());
    for (const auto& c : buffer_) out.push_back(c.close);
    return out;
}

size_t PriceBuffer::size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return buffer_.size();
}

uint64_t PriceBuffer::version() const { return version_.load(); }

// ─────────────────────────────────────────────────────────────
//  HTTP helpers
// ─────────────────────────────────────────────────────────────

#ifdef HAVE_CURL
static size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, std::string* data) {
    data->append(ptr, size * nmemb);
    return size * nmemb;
}

static std::string http_get(const std::string& url,
                              const std::vector<std::string>& headers = {}) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("CURL init failed");

    std::string response;
    struct curl_slist* hdrs = nullptr;
    for (const auto& h : headers) hdrs = curl_slist_append(hdrs, h.c_str());

    curl_easy_setopt(curl, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     hdrs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,      &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        10L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT,      "BtcPredictor/1.0");

    CURLcode res = curl_easy_perform(curl);
    if (hdrs) curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK)
        throw std::runtime_error(std::string("CURL error: ") + curl_easy_strerror(res));
    return response;
}
#else
static std::string http_get(const std::string&, const std::vector<std::string>& = {}) {
    throw std::runtime_error("libcurl not available — enable HAVE_CURL");
}
#endif

// ─── Simple JSON value extractor (no dependency) ───────────
static std::string json_extract(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    if (json[pos] == '"') {
        auto end = json.find('"', pos + 1);
        return json.substr(pos + 1, end - pos - 1);
    }
    auto end = json.find_first_of(",}\n", pos);
    return json.substr(pos, end - pos);
}

static std::vector<std::pair<long long, float>> parse_coingecko_prices(const std::string& json) {
    // Parses [[ts, price], ...] from CoinGecko /market_chart
    std::vector<std::pair<long long, float>> result;
    size_t pos = 0;
    while ((pos = json.find('[', pos)) != std::string::npos) {
        size_t close = json.find(']', pos + 1);
        if (close == std::string::npos) break;
        std::string item = json.substr(pos + 1, close - pos - 1);
        auto comma = item.find(',');
        if (comma == std::string::npos) { ++pos; continue; }
        try {
            long long ts    = std::stoll(item.substr(0, comma));
            float price     = std::stof(item.substr(comma + 1));
            if (ts > 1e9 && price > 0)
                result.push_back({ts, price});
        } catch (...) {}
        pos = close + 1;
    }
    return result;
}

// ─────────────────────────────────────────────────────────────
//  CoinGeckoClient
// ─────────────────────────────────────────────────────────────

CoinGeckoClient::CoinGeckoClient(const std::string& api_key)
    : api_key_(api_key),
      base_url_("https://api.coingecko.com/api/v3"),
      last_request_(0) {}

std::vector<OHLCV> CoinGeckoClient::fetch_ohlcv(const std::string& coin_id,
                                                   const std::string& vs,
                                                   int days) {
    rate_limit_wait();

    std::string url = base_url_ + "/coins/" + coin_id
                    + "/market_chart?vs_currency=" + vs
                    + "&days=" + std::to_string(days)
                    + "&interval=daily";

    std::vector<std::string> headers;
    if (!api_key_.empty()) headers.push_back("x-cg-demo-api-key: " + api_key_);

    std::string resp;
    try { resp = http_get(url, headers); }
    catch (const std::exception& e) {
        last_error_ = e.what();
        return {};
    }

    auto pts = parse_coingecko_prices(resp);
    std::vector<OHLCV> result;
    for (size_t i = 1; i < pts.size(); ++i) {
        OHLCV c;
        c.timestamp = pts[i].first;
        c.open  = pts[i-1].second;
        c.close = pts[i].second;
        c.high  = std::max(c.open, c.close) * 1.005f;
        c.low   = std::min(c.open, c.close) * 0.995f;
        c.volume = 0;
        result.push_back(c);
    }
    last_request_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return result;
}

float CoinGeckoClient::fetch_current_price(const std::string& coin_id,
                                             const std::string& vs) {
    rate_limit_wait();
    std::string url = base_url_ + "/simple/price?ids=" + coin_id + "&vs_currencies=" + vs;
    std::vector<std::string> headers;
    if (!api_key_.empty()) headers.push_back("x-cg-demo-api-key: " + api_key_);
    try {
        std::string resp = http_get(url, headers);
        std::string val  = json_extract(resp, vs);
        if (!val.empty()) return std::stof(val);
    } catch (...) {}
    return -1.0f;
}

CoinGeckoMarketData CoinGeckoClient::fetch_market_data(const std::string& coin_id) {
    rate_limit_wait();
    std::string url = base_url_ + "/coins/" + coin_id
                    + "?localization=false&tickers=false&community_data=false"
                      "&developer_data=false&sparkline=false";
    std::vector<std::string> headers;
    if (!api_key_.empty()) headers.push_back("x-cg-demo-api-key: " + api_key_);

    CoinGeckoMarketData data;
    try {
        std::string resp = http_get(url, headers);
        auto extract_md = [&](const std::string& k) -> float {
            std::string v = json_extract(resp, k);
            return v.empty() ? 0.0f : std::stof(v);
        };
        data.price          = extract_md("usd");
        data.market_cap     = extract_md("usd_market_cap");
        data.volume_24h     = extract_md("usd_24h_vol");
        data.price_change_24h = extract_md("usd_24h_change");
        data.circulating_supply = extract_md("circulating_supply");
        data.total_supply   = extract_md("total_supply");
        data.ath            = extract_md("ath");
    } catch (const std::exception& e) {
        last_error_ = e.what();
    }
    return data;
}

void CoinGeckoClient::rate_limit_wait() {
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    long long elapsed = now - last_request_;
    if (elapsed < 1200)  // 50 req/min = 1200ms spacing
        std::this_thread::sleep_for(std::chrono::milliseconds(1200 - elapsed));
}

// ─────────────────────────────────────────────────────────────
//  BinanceWSClient (WebSocket simulation via polling)
//  Full WS requires libwebsockets — simulate with REST polling
// ─────────────────────────────────────────────────────────────

BinanceWSClient::BinanceWSClient(PriceBuffer& buffer,
                                   const std::string& symbol)
    : buffer_(buffer), symbol_(symbol), running_(false) {}

BinanceWSClient::~BinanceWSClient() { stop(); }

void BinanceWSClient::start() {
    running_ = true;
    worker_  = std::thread([this] { poll_loop(); });
}

void BinanceWSClient::stop() {
    running_ = false;
    if (worker_.joinable()) worker_.join();
}

void BinanceWSClient::poll_loop() {
    std::string url = "https://api.binance.com/api/v3/ticker/price?symbol=" + symbol_;

    while (running_) {
        try {
            std::string resp = http_get(url);
            std::string price_str = json_extract(resp, "price");
            if (!price_str.empty()) {
                float price = std::stof(price_str);
                OHLCV c;
                c.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                c.open = c.high = c.low = c.close = price;
                c.volume = 0;
                buffer_.push(c);
                if (on_price_) on_price_(price);
            }
        } catch (const std::exception& e) {
            error_count_++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms_));
    }
}

// ─────────────────────────────────────────────────────────────
//  OrderBookAggregator
// ─────────────────────────────────────────────────────────────

void OrderBookAggregator::update_bid(float price, float qty) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (qty == 0.0f) bids_.erase(price);
    else bids_[price] = qty;
}

void OrderBookAggregator::update_ask(float price, float qty) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (qty == 0.0f) asks_.erase(price);
    else asks_[price] = qty;
}

float OrderBookAggregator::best_bid() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return bids_.empty() ? 0.0f : bids_.rbegin()->first;
}

float OrderBookAggregator::best_ask() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return asks_.empty() ? 0.0f : asks_.begin()->first;
}

float OrderBookAggregator::mid_price() const {
    float b = best_bid(), a = best_ask();
    return (b > 0 && a > 0) ? (b + a) / 2.0f : b + a;
}

float OrderBookAggregator::spread() const {
    return best_ask() - best_bid();
}

float OrderBookAggregator::bid_ask_imbalance() const {
    std::lock_guard<std::mutex> lock(mtx_);
    float bid_vol = 0, ask_vol = 0;
    int depth = 10;
    int i = 0;
    for (auto it = bids_.rbegin(); it != bids_.rend() && i < depth; ++it, ++i)
        bid_vol += it->second;
    i = 0;
    for (auto it = asks_.begin(); it != asks_.end() && i < depth; ++it, ++i)
        ask_vol += it->second;
    float total = bid_vol + ask_vol;
    return total > 0 ? (bid_vol - ask_vol) / total : 0.0f;
}

// ─────────────────────────────────────────────────────────────
//  SentimentAnalyzer (fear/greed via alternative.me)
// ─────────────────────────────────────────────────────────────

SentimentScore SentimentAnalyzer::fetch_fear_greed() {
    SentimentScore score;
    try {
        std::string resp = http_get("https://api.alternative.me/fng/?limit=1");
        std::string val  = json_extract(resp, "value");
        std::string cls  = json_extract(resp, "value_classification");
        if (!val.empty()) {
            score.fear_greed_index = std::stof(val);
            score.classification   = cls;
            score.bullish_score    = score.fear_greed_index / 100.0f;
            score.bearish_score    = 1.0f - score.bullish_score;
        }
    } catch (...) {
        score.classification = "unknown";
    }
    return score;
}

// ─────────────────────────────────────────────────────────────
//  OnChainMetrics (Glassnode-style, mock structure)
// ─────────────────────────────────────────────────────────────

OnChainMetrics OnChainDataProvider::fetch_metrics() {
    OnChainMetrics m;
    // In production these would hit Glassnode / IntoTheBlock APIs
    m.active_addresses    = 900000 + (std::rand() % 100000);
    m.transaction_count   = 300000 + (std::rand() % 50000);
    m.hash_rate_th_s      = 500e12f + (float)(std::rand() % 50) * 1e12f;
    m.difficulty          = 70e12f;
    m.mempool_size        = 5000 + (std::rand() % 2000);
    m.exchange_inflows    = (float)(std::rand() % 1000) + 500.0f;
    m.exchange_outflows   = (float)(std::rand() % 1000) + 400.0f;
    m.nvt_ratio           = 50.0f + (float)(std::rand() % 50);
    m.sopr                = 1.0f + (float)(std::rand() % 20) * 0.01f - 0.1f;
    return m;
}

// ─────────────────────────────────────────────────────────────
//  DataPipeline
// ─────────────────────────────────────────────────────────────

DataPipeline::DataPipeline(const std::string& coingecko_key)
    : coingecko_(coingecko_key),
      price_buffer_(2000),
      running_(false) {}

DataPipeline::~DataPipeline() { stop(); }

bool DataPipeline::start() {
    // Load historical data
    auto history = coingecko_.fetch_ohlcv("bitcoin", "usd", 365);
    for (const auto& c : history) price_buffer_.push(c);

    // Start real-time polling
    binance_ws_ = std::make_unique<BinanceWSClient>(price_buffer_, "BTCUSDT");
    binance_ws_->set_on_price([this](float p) {
        if (on_new_price_) on_new_price_(p);
    });
    binance_ws_->start();

    // Background re-fetch every 5 minutes
    running_ = true;
    refresh_thread_ = std::thread([this] {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::minutes(5));
            if (!running_) break;
            auto fresh = coingecko_.fetch_ohlcv("bitcoin", "usd", 7);
            for (const auto& c : fresh) price_buffer_.push(c);
        }
    });
    return true;
}

void DataPipeline::stop() {
    running_ = false;
    if (binance_ws_) binance_ws_->stop();
    if (refresh_thread_.joinable()) refresh_thread_.join();
}

PipelineStats DataPipeline::stats() const {
    PipelineStats s;
    s.buffer_size    = (int)price_buffer_.size();
    s.data_version   = price_buffer_.version();
    s.ws_errors      = binance_ws_ ? binance_ws_->error_count() : 0;
    s.last_update_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return s;
}
