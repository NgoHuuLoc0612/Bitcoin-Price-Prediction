/**
 * core.hpp — Bitcoin Price Prediction Engine v3
 * Models: LSTM · Transformer · TCN · WaveNet · N-BEATS · Informer · NHiTS · TFT
 *         PatchTST · TimesNet · DLinear · Crossformer  (12 total)
 * SIMD AVX2 · OpenBLAS · NUMA-aware · Monte-Carlo uncertainty (10 passes)
 */

#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <string>
#include <functional>
#include <unordered_map>
#include <array>

// ─────────────────────────────────────────────────────────────
//  ModelConfig
// ─────────────────────────────────────────────────────────────

struct ModelConfig {
    int input_size          = 4;
    int hidden_size         = 64;
    int num_lstm_layers     = 3;
    int num_heads           = 4;
    int num_tcn_layers      = 4;
    int tcn_kernel_size     = 3;
    // WaveNet
    int wavenet_layers      = 8;
    int wavenet_residual_ch = 32;
    int wavenet_skip_ch     = 64;
    // N-BEATS
    int nbeats_stacks       = 3;
    int nbeats_blocks       = 4;
    int nbeats_hidden       = 128;
    // Informer
    int informer_factor     = 5;
    int informer_d_ff       = 256;
    int informer_enc_layers = 3;
    // NHiTS
    int nhits_stacks        = 3;
    int nhits_hidden        = 256;
    std::vector<int> nhits_pool_sizes{4, 2, 1};
    // TFT
    int tft_hidden          = 64;
    int tft_num_heads       = 4;
    // PatchTST
    int patchtst_patch_len  = 16;
    int patchtst_stride     = 8;
    int patchtst_d_model    = 64;
    int patchtst_n_heads    = 4;
    int patchtst_num_layers = 3;
    // TimesNet
    int timesnet_d_model    = 64;
    int timesnet_d_ff       = 256;
    int timesnet_num_layers = 2;
    int timesnet_top_k      = 5;
    // DLinear
    int dlinear_moving_avg  = 25;
    // Crossformer
    int crossformer_seg_len  = 6;
    int crossformer_d_model  = 64;
    int crossformer_n_heads  = 4;
    int crossformer_num_layers= 2;
    // Shared
    int sequence_length     = 60;
    int forecast_steps      = 10;
    float dropout_rate      = 0.2f;
    float learning_rate     = 1e-3f;
    int batch_size          = 32;
    int max_epochs          = 200;
    // Ensemble weights [12]: LSTM Transformer TCN WaveNet NBEATS
    //                        Informer NHiTS TFT PatchTST TimesNet DLinear Crossformer
    std::array<float, 12> ensemble_weights{
        0.10f, 0.09f, 0.09f, 0.09f, 0.08f, 0.08f,
        0.08f, 0.08f, 0.10f, 0.09f, 0.06f, 0.06f
    };
};

// ─────────────────────────────────────────────────────────────
//  Tensor2D
// ─────────────────────────────────────────────────────────────

struct Tensor2D {
    size_t rows, cols;
    std::vector<float> data;

    Tensor2D();
    Tensor2D(size_t r, size_t c, float val = 0.0f);
    float&       at(size_t r, size_t c);
    const float& at(size_t r, size_t c) const;

    Tensor2D matmul(const Tensor2D& other) const;
    Tensor2D transpose() const;
    Tensor2D elementwise_mul(const Tensor2D& other) const;
    Tensor2D add(const Tensor2D& other) const;
    Tensor2D sub(const Tensor2D& other) const;
    void apply_sigmoid();
    void apply_tanh();
    void apply_relu();
    void apply_gelu();
    void layer_norm(float eps = 1e-6f);
    Tensor2D slice_rows(size_t start, size_t end) const;
    Tensor2D concat_rows(const Tensor2D& other) const;
};

// ─────────────────────────────────────────────────────────────
//  LSTM
// ─────────────────────────────────────────────────────────────

struct LSTMState { Tensor2D h, c; };

struct LSTMCell {
    int input_size, hidden_size;
    Tensor2D Wf, Wi, Wc, Wo, bf, bi, bc, bo;
    LSTMCell(int input_size, int hidden_size);
    LSTMState forward(const Tensor2D& x, const LSTMState& prev);
};

// ─────────────────────────────────────────────────────────────
//  Transformer
// ─────────────────────────────────────────────────────────────

struct MultiHeadAttention {
    int d_model, n_heads, d_k;
    Tensor2D Wq, Wk, Wv, Wo;
    MultiHeadAttention(int d_model, int n_heads);
    Tensor2D forward(const Tensor2D& x);
    Tensor2D scaled_dot_product(const Tensor2D& Q, const Tensor2D& K, const Tensor2D& V);
    // Cross-attention variant: query from q, key/value from kv
    Tensor2D cross_forward(const Tensor2D& q, const Tensor2D& kv);
};

struct TransformerBlock {
    std::unique_ptr<MultiHeadAttention> attn;
    Tensor2D ffn_W1, ffn_b1, ffn_W2, ffn_b2;
    int d_model, d_ff;
    TransformerBlock(int d_model, int n_heads, int d_ff);
    Tensor2D forward(const Tensor2D& x);
};

// ─────────────────────────────────────────────────────────────
//  TCN
// ─────────────────────────────────────────────────────────────

struct TCNBlock {
    int channels, kernel_size, dilation;
    Tensor2D W1, W2, b1, b2;
    TCNBlock(int channels, int kernel_size, int dilation);
    Tensor2D forward(const Tensor2D& x);
};

// ─────────────────────────────────────────────────────────────
//  WaveNet
// ─────────────────────────────────────────────────────────────

struct WaveNetLayer {
    int residual_ch, skip_ch, kernel_size, dilation;
    Tensor2D W_filter, W_gate, b_filter, b_gate;
    Tensor2D W_res, b_res;
    Tensor2D W_skip, b_skip;
    WaveNetLayer(int residual_ch, int skip_ch, int kernel_size, int dilation);
    std::pair<Tensor2D, Tensor2D> forward(const Tensor2D& x);
};

struct WaveNet {
    int num_layers, residual_ch, skip_ch, hidden;
    Tensor2D W_in, b_in;
    Tensor2D W_out1, b_out1, W_out2, b_out2;
    std::vector<WaveNetLayer> layers;
    WaveNet(int num_layers, int residual_ch, int skip_ch, int hidden);
    float forward(const std::vector<float>& seq);
};

// ─────────────────────────────────────────────────────────────
//  N-BEATS
// ─────────────────────────────────────────────────────────────

enum class NBEATSBlockType { TREND, SEASONALITY, GENERIC };

struct NBEATSBlock {
    NBEATSBlockType block_type;
    int seq_len, forecast_len, hidden, degree;
    std::vector<Tensor2D> fc_W, fc_b;
    Tensor2D theta_b_W, theta_f_W;
    Tensor2D basis_b, basis_f;

    NBEATSBlock(NBEATSBlockType type, int seq_len, int forecast_len, int hidden, int degree);
    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<float>& x);

private:
    void build_trend_basis(int s, int f, int deg);
    void build_seasonality_basis(int s, int f);
    std::vector<float> fc_forward(const std::vector<float>& x);
    std::vector<float> matmul_vec(const Tensor2D& W, const std::vector<float>& x);
};

struct NBEATSStack {
    std::vector<NBEATSBlock> blocks;
    NBEATSStack(NBEATSBlockType type, int num_blocks,
                int seq_len, int forecast_len, int hidden, int degree);
    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<float>& x);
};

struct NBEATS {
    std::vector<NBEATSStack> stacks;
    int seq_len, forecast_len;
    NBEATS(int seq_len, int forecast_len, int hidden, int num_stacks, int blocks_per_stack);
    float forward(const std::vector<float>& seq);
};

// ─────────────────────────────────────────────────────────────
//  Informer
// ─────────────────────────────────────────────────────────────

struct ProbSparseAttention {
    int d_model, n_heads, d_k, factor;
    Tensor2D Wq, Wk, Wv, Wo;
    ProbSparseAttention(int d_model, int n_heads, int factor = 5);
    Tensor2D forward(const Tensor2D& Q, const Tensor2D& K, const Tensor2D& V);
};

struct InformerEncoderLayer {
    std::unique_ptr<ProbSparseAttention> attn;
    Tensor2D ffn_W1, ffn_b1, ffn_W2, ffn_b2;
    int d_model, d_ff;
    InformerEncoderLayer(int d_model, int n_heads, int d_ff, int factor);
    Tensor2D forward(const Tensor2D& x, bool distill = true);
};

struct Informer {
    int d_model, enc_layers;
    Tensor2D W_embed, b_embed;
    std::vector<InformerEncoderLayer> encoder_layers;
    Tensor2D W_proj, b_proj;
    Informer(int d_model, int n_heads, int d_ff, int enc_layers, int factor);
    float forward(const std::vector<float>& seq);
};

// ─────────────────────────────────────────────────────────────
//  NHiTS
// ─────────────────────────────────────────────────────────────

struct NHiTSBlock {
    int input_size, output_size, hidden, pool_size;
    std::vector<Tensor2D> mlp_W, mlp_b;
    Tensor2D W_backcast, b_backcast, W_forecast, b_forecast;

    NHiTSBlock(int input_size, int output_size, int hidden, int pool_size);
    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<float>& x);

private:
    std::vector<float> max_pool(const std::vector<float>& x, int p);
    std::vector<float> linear_interp(const std::vector<float>& x, int target_len);
    std::vector<float> mlp_forward(const std::vector<float>& x);
    std::vector<float> matmul_vec(const Tensor2D& W, const Tensor2D& b, const std::vector<float>& x);
};

struct NHiTSStack {
    std::vector<NHiTSBlock> blocks;
    NHiTSStack(int input_size, int output_size, int hidden, int num_blocks, int pool_size);
    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<float>& x);
};

struct NHiTS {
    std::vector<NHiTSStack> stacks;
    int seq_len, forecast_len;
    NHiTS(int seq_len, int forecast_len, int hidden, int num_stacks,
          const std::vector<int>& pool_sizes);
    float forward(const std::vector<float>& seq);
};

// ─────────────────────────────────────────────────────────────
//  TFT
// ─────────────────────────────────────────────────────────────

struct GatedResidualNetwork {
    int d_in, d_hidden, d_out;
    Tensor2D W1, b1, W2, b2;
    Tensor2D Wg1, bg1, Wg2, bg2;
    Tensor2D W_skip, b_skip;
    bool has_skip;
    GatedResidualNetwork(int d_in, int d_hidden, int d_out);
    std::vector<float> forward(const std::vector<float>& x);
private:
    std::vector<float> matmul_vec(const Tensor2D& W, const Tensor2D& b, const std::vector<float>& x);
};

struct VariableSelectionNetwork {
    int num_vars, d_model;
    std::vector<GatedResidualNetwork> var_grns;
    GatedResidualNetwork flat_grn;
    Tensor2D W_softmax;
    VariableSelectionNetwork(int num_vars, int d_model);
    std::vector<float> forward(const std::vector<std::vector<float>>& vars);
};

struct TFT {
    int d_model, seq_len;
    Tensor2D W_embed, b_embed;
    std::unique_ptr<VariableSelectionNetwork> vsn;
    std::unique_ptr<MultiHeadAttention> temporal_attn;
    GatedResidualNetwork post_attn_grn;
    Tensor2D W_q10, W_q50, W_q90, b_q10, b_q50, b_q90;
    TFT(int d_model, int n_heads, int seq_len);
    std::array<float, 3> forward(const std::vector<float>& seq);
};

// ─────────────────────────────────────────────────────────────
//  PatchTST — patch-based channel-independent Transformer (2023)
// ─────────────────────────────────────────────────────────────

struct PatchTSTLayer {
    int d_model, d_ff;
    std::unique_ptr<MultiHeadAttention> attn;
    Tensor2D ffn_W1, ffn_b1, ffn_W2, ffn_b2;
    PatchTSTLayer(int d_model, int n_heads, int d_ff);
    Tensor2D forward(const Tensor2D& x);
};

struct PatchTST {
    int patch_len, stride, d_model, seq_len, num_patches;
    Tensor2D W_patch, b_patch;   // patch_len -> d_model projection
    std::vector<PatchTSTLayer> enc_layers;
    Tensor2D W_head, b_head;     // num_patches*d_model -> 1
    PatchTST(int seq_len, int patch_len, int stride, int d_model, int n_heads, int num_layers);
    float forward(const std::vector<float>& seq);
private:
    std::vector<std::vector<float>> extract_patches(const std::vector<float>& seq) const;
};

// ─────────────────────────────────────────────────────────────
//  TimesNet — FFT-guided 2D temporal variation modeling (2023)
// ─────────────────────────────────────────────────────────────

struct TimesBlock {
    int d_model, d_ff, top_k;
    Tensor2D W_conv_p, b_conv_p;  // periodic-dim mixing
    Tensor2D W_conv_t, b_conv_t;  // temporal-dim mixing
    Tensor2D ffn_W1, ffn_b1, ffn_W2, ffn_b2;
    TimesBlock(int d_model, int d_ff, int top_k);
    Tensor2D forward(const Tensor2D& x);
private:
    std::vector<int> fft_top_periods(const std::vector<float>& signal) const;
};

struct TimesNet {
    int d_model, num_layers, seq_len;
    Tensor2D W_embed, b_embed;
    std::vector<TimesBlock> blocks;
    Tensor2D W_proj, b_proj;
    TimesNet(int seq_len, int d_model, int d_ff, int num_layers, int top_k);
    float forward(const std::vector<float>& seq);
};

// ─────────────────────────────────────────────────────────────
//  DLinear — moving-average decomposition + dual linear layers (2023)
// ─────────────────────────────────────────────────────────────

struct DLinear {
    int seq_len, forecast_len, moving_avg;
    Tensor2D W_trend, b_trend;       // seq_len -> forecast_len
    Tensor2D W_seasonal, b_seasonal; // seq_len -> forecast_len
    DLinear(int seq_len, int forecast_len, int moving_avg);
    float forward(const std::vector<float>& seq);
private:
    std::vector<float> moving_avg_filter(const std::vector<float>& x, int k) const;
    float linear_proj(const Tensor2D& W, const Tensor2D& b, const std::vector<float>& x) const;
};

// ─────────────────────────────────────────────────────────────
//  Crossformer — two-stage cross-time/cross-dim attention (2023)
// ─────────────────────────────────────────────────────────────

struct CrossformerLayer {
    int d_model, d_ff;
    std::unique_ptr<MultiHeadAttention> time_attn;
    std::unique_ptr<MultiHeadAttention> dim_attn;
    Tensor2D ffn_W1, ffn_b1, ffn_W2, ffn_b2;
    CrossformerLayer(int d_model, int n_heads, int d_ff);
    Tensor2D forward(const Tensor2D& x);
};

struct Crossformer {
    int seg_len, d_model, seq_len, num_segs;
    Tensor2D W_seg, b_seg;
    std::vector<CrossformerLayer> enc_layers;
    Tensor2D W_proj, b_proj;
    Crossformer(int seq_len, int seg_len, int d_model, int n_heads, int num_layers);
    float forward(const std::vector<float>& seq);
private:
    std::vector<std::vector<float>> segment(const std::vector<float>& seq) const;
};

// ─────────────────────────────────────────────────────────────
//  PredictionResult — 12 model outputs + TFT quantiles + ensemble
// ─────────────────────────────────────────────────────────────

struct PredictionResult {
    float lstm_pred        = 0.0f;
    float transformer_pred = 0.0f;
    float tcn_pred         = 0.0f;
    float wavenet_pred     = 0.0f;
    float nbeats_pred      = 0.0f;
    float informer_pred    = 0.0f;
    float nhits_pred       = 0.0f;
    float tft_pred         = 0.0f;
    float tft_q10          = 0.0f;
    float tft_q90          = 0.0f;
    float patchtst_pred    = 0.0f;
    float timesnet_pred    = 0.0f;
    float dlinear_pred     = 0.0f;
    float crossformer_pred = 0.0f;
    float ensemble_pred    = 0.0f;
    float confidence       = 0.0f;
    float uncertainty      = 0.0f;
    long long timestamp    = 0;
};

struct ModelMetrics {
    float rmse               = 0.0f;
    float mae                = 0.0f;
    float mape               = 0.0f;
    float r2                 = 0.0f;
    float sharpe_ratio       = 0.0f;
    float directional_accuracy = 0.0f;
    int   data_points        = 0;
    bool  is_trained         = false;
    std::array<float, 12> per_model_rmse{};
    std::array<float, 12> ensemble_weights{};
};

// ─────────────────────────────────────────────────────────────
//  EnsemblePredictor — 12 models, adaptive inverse-error weighting
// ─────────────────────────────────────────────────────────────

struct EnsemblePredictor {
    ModelConfig config;
    // 8 original models
    std::vector<LSTMCell>              lstm_layers;
    std::unique_ptr<TransformerBlock>  transformer_block;
    std::vector<TCNBlock>              tcn_blocks;
    std::unique_ptr<WaveNet>           wavenet;
    std::unique_ptr<NBEATS>            nbeats;
    std::unique_ptr<Informer>          informer;
    std::unique_ptr<NHiTS>             nhits;
    std::unique_ptr<TFT>               tft;
    // 4 new models
    std::unique_ptr<PatchTST>          patchtst;
    std::unique_ptr<TimesNet>          timesnet;
    std::unique_ptr<DLinear>           dlinear;
    std::unique_ptr<Crossformer>       crossformer;

    Tensor2D fc_out, fc_bias;
    std::array<float, 12> model_weights;
    std::array<float, 12> model_error_ema;

    explicit EnsemblePredictor(const ModelConfig& cfg);
    PredictionResult predict(const std::vector<float>& sequence);
    void update_weights(const std::array<float, 12>& errors);

private:
    float predict_lstm(const std::vector<float>& seq);
    float predict_transformer(const std::vector<float>& seq);
    float predict_tcn(const std::vector<float>& seq);
    float predict_wavenet(const std::vector<float>& seq);
    float predict_nbeats(const std::vector<float>& seq);
    float predict_informer(const std::vector<float>& seq);
    float predict_nhits(const std::vector<float>& seq);
    std::array<float,3> predict_tft(const std::vector<float>& seq);
    float predict_patchtst(const std::vector<float>& seq);
    float predict_timesnet(const std::vector<float>& seq);
    float predict_dlinear(const std::vector<float>& seq);
    float predict_crossformer(const std::vector<float>& seq);
};

// ─────────────────────────────────────────────────────────────
//  Feature Engineering — 10 technical indicators
// ─────────────────────────────────────────────────────────────

struct FeatureEngineering {
    static std::vector<float> compute_rsi(const std::vector<float>& prices, int period = 14);
    static std::vector<float> compute_macd(const std::vector<float>& prices,
                                            int fast = 12, int slow = 26, int signal = 9);
    static std::vector<float> compute_bollinger(const std::vector<float>& prices, int period = 20);
    static std::vector<float> compute_atr(const std::vector<float>& prices, int period = 14);
    static std::vector<float> compute_williams_r(const std::vector<float>& prices, int period = 14);
    static std::vector<float> compute_cci(const std::vector<float>& prices, int period = 20);
    static std::vector<float> compute_stochastic_k(const std::vector<float>& prices, int k = 14);
    static std::vector<float> compute_log_returns(const std::vector<float>& prices);
    static std::vector<float> compute_realised_vol(const std::vector<float>& returns, int window = 20);
    static std::vector<std::vector<float>> build_feature_matrix(
        const std::vector<float>& prices, const std::vector<float>& volumes = {});
};

// ─────────────────────────────────────────────────────────────
//  DataNormalizer — z-score & robust (median/IQR)
// ─────────────────────────────────────────────────────────────

struct DataNormalizer {
    float mean_ = 0.0f, std_ = 1.0f;
    float median_ = 0.0f, iqr_ = 1.0f;
    bool fitted_ = false, robust_mode = false;

    void fit(const std::vector<float>& data, bool robust = false);
    std::vector<float> transform(const std::vector<float>& data) const;
    std::vector<float> inverse_transform(const std::vector<float>& data) const;
    float inverse_scalar(float v) const;
};

// ─────────────────────────────────────────────────────────────
//  RealTimePredictor
// ─────────────────────────────────────────────────────────────

class RealTimePredictor {
public:
    explicit RealTimePredictor(const ModelConfig& cfg);
    void push_price(float price);
    void push_prices(const std::vector<float>& prices);
    PredictionResult predict_next();
    std::vector<PredictionResult> forecast_horizon(int steps);
    ModelMetrics get_metrics() const;
    std::string serialize_weights() const;
    bool trained() const { return is_trained.load(); }
    float get_loss() const { return last_train_loss; }
    int buffer_size() const { return (int)price_buffer.size(); }

private:
    EnsemblePredictor ensemble;
    DataNormalizer normalizer;
    ModelConfig config;
    std::vector<float> price_buffer;
    mutable std::mutex price_mutex;
    std::atomic<bool>  is_trained{false};
    float last_train_loss = 1.0f;
    int   train_tick      = 0;
    std::array<float, 12> per_model_errors{};
    mutable PredictionResult last_result;

    void train_async();
    void adaptive_weight_update(const std::vector<std::pair<std::vector<float>,float>>& windows);
};
