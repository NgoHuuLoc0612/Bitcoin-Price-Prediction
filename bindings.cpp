/**
 * bindings.cpp — pybind11 Python bindings for btc_engine v3
 * Exposes all 12 models, PredictionResult, ModelMetrics, BacktestEngine, etc.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core.hpp"
#include "optimization.hpp"
#include "external.hpp"

namespace py = pybind11;

// Helper: convert Tensor2D ↔ numpy array
static Tensor2D from_numpy(py::array_t<float> arr) {
    auto buf = arr.request();
    if (buf.ndim == 1) {
        Tensor2D t(1, buf.shape[0]);
        std::memcpy(t.data.data(), buf.ptr, buf.size * sizeof(float));
        return t;
    }
    Tensor2D t(buf.shape[0], buf.shape[1]);
    std::memcpy(t.data.data(), buf.ptr, buf.size * sizeof(float));
    return t;
}

static py::array_t<float> to_numpy(const Tensor2D& t) {
    py::array_t<float> arr({(py::ssize_t)t.rows, (py::ssize_t)t.cols});
    std::memcpy(arr.mutable_data(), t.data.data(), t.data.size() * sizeof(float));
    return arr;
}

PYBIND11_MODULE(btc_engine, m) {
    m.doc() = "Bitcoin Price Prediction Engine v3 — 12 ML models, AVX2 SIMD, NUMA-aware";

    // ── ModelConfig ───────────────────────────────────────────────
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("input_size",            &ModelConfig::input_size)
        .def_readwrite("hidden_size",           &ModelConfig::hidden_size)
        .def_readwrite("num_lstm_layers",       &ModelConfig::num_lstm_layers)
        .def_readwrite("num_heads",             &ModelConfig::num_heads)
        .def_readwrite("num_tcn_layers",        &ModelConfig::num_tcn_layers)
        .def_readwrite("tcn_kernel_size",       &ModelConfig::tcn_kernel_size)
        .def_readwrite("wavenet_layers",        &ModelConfig::wavenet_layers)
        .def_readwrite("wavenet_residual_ch",   &ModelConfig::wavenet_residual_ch)
        .def_readwrite("wavenet_skip_ch",       &ModelConfig::wavenet_skip_ch)
        .def_readwrite("nbeats_stacks",         &ModelConfig::nbeats_stacks)
        .def_readwrite("nbeats_blocks",         &ModelConfig::nbeats_blocks)
        .def_readwrite("nbeats_hidden",         &ModelConfig::nbeats_hidden)
        .def_readwrite("informer_factor",       &ModelConfig::informer_factor)
        .def_readwrite("informer_d_ff",         &ModelConfig::informer_d_ff)
        .def_readwrite("informer_enc_layers",   &ModelConfig::informer_enc_layers)
        .def_readwrite("nhits_stacks",          &ModelConfig::nhits_stacks)
        .def_readwrite("nhits_hidden",          &ModelConfig::nhits_hidden)
        .def_readwrite("nhits_pool_sizes",      &ModelConfig::nhits_pool_sizes)
        .def_readwrite("tft_hidden",            &ModelConfig::tft_hidden)
        .def_readwrite("tft_num_heads",         &ModelConfig::tft_num_heads)
        .def_readwrite("patchtst_patch_len",    &ModelConfig::patchtst_patch_len)
        .def_readwrite("patchtst_stride",       &ModelConfig::patchtst_stride)
        .def_readwrite("patchtst_d_model",      &ModelConfig::patchtst_d_model)
        .def_readwrite("patchtst_n_heads",      &ModelConfig::patchtst_n_heads)
        .def_readwrite("patchtst_num_layers",   &ModelConfig::patchtst_num_layers)
        .def_readwrite("timesnet_d_model",      &ModelConfig::timesnet_d_model)
        .def_readwrite("timesnet_d_ff",         &ModelConfig::timesnet_d_ff)
        .def_readwrite("timesnet_num_layers",   &ModelConfig::timesnet_num_layers)
        .def_readwrite("timesnet_top_k",        &ModelConfig::timesnet_top_k)
        .def_readwrite("dlinear_moving_avg",    &ModelConfig::dlinear_moving_avg)
        .def_readwrite("crossformer_seg_len",   &ModelConfig::crossformer_seg_len)
        .def_readwrite("crossformer_d_model",   &ModelConfig::crossformer_d_model)
        .def_readwrite("crossformer_n_heads",   &ModelConfig::crossformer_n_heads)
        .def_readwrite("crossformer_num_layers",&ModelConfig::crossformer_num_layers)
        .def_readwrite("sequence_length",       &ModelConfig::sequence_length)
        .def_readwrite("forecast_steps",        &ModelConfig::forecast_steps)
        .def_readwrite("dropout_rate",          &ModelConfig::dropout_rate)
        .def_readwrite("learning_rate",         &ModelConfig::learning_rate)
        .def_readwrite("batch_size",            &ModelConfig::batch_size)
        .def_readwrite("max_epochs",            &ModelConfig::max_epochs)
        .def_property("ensemble_weights",
            [](const ModelConfig& c) {
                return std::vector<float>(c.ensemble_weights.begin(), c.ensemble_weights.end());
            },
            [](ModelConfig& c, const std::vector<float>& v) {
                for (int i=0;i<12&&i<(int)v.size();++i) c.ensemble_weights[i]=v[i];
            });

    // ── PredictionResult ─────────────────────────────────────────
    py::class_<PredictionResult>(m, "PredictionResult")
        .def(py::init<>())
        .def_readwrite("lstm_pred",        &PredictionResult::lstm_pred)
        .def_readwrite("transformer_pred", &PredictionResult::transformer_pred)
        .def_readwrite("tcn_pred",         &PredictionResult::tcn_pred)
        .def_readwrite("wavenet_pred",     &PredictionResult::wavenet_pred)
        .def_readwrite("nbeats_pred",      &PredictionResult::nbeats_pred)
        .def_readwrite("informer_pred",    &PredictionResult::informer_pred)
        .def_readwrite("nhits_pred",       &PredictionResult::nhits_pred)
        .def_readwrite("tft_pred",         &PredictionResult::tft_pred)
        .def_readwrite("tft_q10",          &PredictionResult::tft_q10)
        .def_readwrite("tft_q90",          &PredictionResult::tft_q90)
        .def_readwrite("patchtst_pred",    &PredictionResult::patchtst_pred)
        .def_readwrite("timesnet_pred",    &PredictionResult::timesnet_pred)
        .def_readwrite("dlinear_pred",     &PredictionResult::dlinear_pred)
        .def_readwrite("crossformer_pred", &PredictionResult::crossformer_pred)
        .def_readwrite("ensemble_pred",    &PredictionResult::ensemble_pred)
        .def_readwrite("confidence",       &PredictionResult::confidence)
        .def_readwrite("uncertainty",      &PredictionResult::uncertainty)
        .def_readwrite("timestamp",        &PredictionResult::timestamp)
        .def("to_dict", [](const PredictionResult& r) {
            py::dict d;
            d["lstm"]        = r.lstm_pred;
            d["transformer"] = r.transformer_pred;
            d["tcn"]         = r.tcn_pred;
            d["wavenet"]     = r.wavenet_pred;
            d["nbeats"]      = r.nbeats_pred;
            d["informer"]    = r.informer_pred;
            d["nhits"]       = r.nhits_pred;
            d["tft"]         = r.tft_pred;
            d["tft_q10"]     = r.tft_q10;
            d["tft_q90"]     = r.tft_q90;
            d["patchtst"]    = r.patchtst_pred;
            d["timesnet"]    = r.timesnet_pred;
            d["dlinear"]     = r.dlinear_pred;
            d["crossformer"] = r.crossformer_pred;
            d["ensemble"]    = r.ensemble_pred;
            d["confidence"]  = r.confidence;
            d["uncertainty"] = r.uncertainty;
            d["timestamp"]   = r.timestamp;
            return d;
        });

    // ── ModelMetrics ─────────────────────────────────────────────
    py::class_<ModelMetrics>(m, "ModelMetrics")
        .def(py::init<>())
        .def_readwrite("rmse",                &ModelMetrics::rmse)
        .def_readwrite("mae",                 &ModelMetrics::mae)
        .def_readwrite("mape",                &ModelMetrics::mape)
        .def_readwrite("r2",                  &ModelMetrics::r2)
        .def_readwrite("sharpe_ratio",        &ModelMetrics::sharpe_ratio)
        .def_readwrite("directional_accuracy",&ModelMetrics::directional_accuracy)
        .def_readwrite("data_points",         &ModelMetrics::data_points)
        .def_readwrite("is_trained",          &ModelMetrics::is_trained)
        .def_property_readonly("per_model_rmse", [](const ModelMetrics& m) {
            return std::vector<float>(m.per_model_rmse.begin(), m.per_model_rmse.end());
        })
        .def_property_readonly("ensemble_weights", [](const ModelMetrics& m) {
            return std::vector<float>(m.ensemble_weights.begin(), m.ensemble_weights.end());
        });

    // ── TrainConfig ──────────────────────────────────────────────
    py::class_<TrainConfig>(m, "TrainConfig")
        .def(py::init<>())
        .def_readwrite("epochs",       &TrainConfig::epochs)
        .def_readwrite("batch_size",   &TrainConfig::batch_size)
        .def_readwrite("lr",           &TrainConfig::lr)
        .def_readwrite("patience",     &TrainConfig::patience)
        .def_readwrite("clip_norm",    &TrainConfig::clip_norm)
        .def_readwrite("weight_decay", &TrainConfig::weight_decay);

    // ── TrainResult ──────────────────────────────────────────────
    py::class_<TrainResult>(m, "TrainResult")
        .def(py::init<>())
        .def_readwrite("final_loss",    &TrainResult::final_loss)
        .def_readwrite("best_loss",     &TrainResult::best_loss)
        .def_readwrite("epochs_run",    &TrainResult::epochs_run)
        .def_readwrite("converged",     &TrainResult::converged)
        .def_readwrite("loss_history",  &TrainResult::loss_history);

    // ── BacktestResult ───────────────────────────────────────────
    py::class_<BacktestResult>(m, "BacktestResult")
        .def(py::init<>())
        .def_readwrite("initial_capital", &BacktestResult::initial_capital)
        .def_readwrite("final_capital",   &BacktestResult::final_capital)
        .def_readwrite("total_return",    &BacktestResult::total_return)
        .def_readwrite("max_drawdown",    &BacktestResult::max_drawdown)
        .def_readwrite("sharpe_ratio",    &BacktestResult::sharpe_ratio)
        .def_readwrite("win_rate",        &BacktestResult::win_rate)
        .def_readwrite("num_trades",      &BacktestResult::num_trades)
        .def_readwrite("equity_curve",    &BacktestResult::equity_curve);

    // ── RealTimePredictor ────────────────────────────────────────
    py::class_<RealTimePredictor>(m, "RealTimePredictor")
        .def(py::init<const ModelConfig&>())
        .def("push_price",  &RealTimePredictor::push_price)
        .def("push_prices", &RealTimePredictor::push_prices)
        .def("predict_next",&RealTimePredictor::predict_next)
        .def("forecast_horizon", &RealTimePredictor::forecast_horizon)
        .def("get_metrics", &RealTimePredictor::get_metrics)
        .def("serialize_weights", &RealTimePredictor::serialize_weights)
        .def("trained",     &RealTimePredictor::trained)
        .def("get_loss",    &RealTimePredictor::get_loss)
        .def("buffer_size", &RealTimePredictor::buffer_size);

    // ── FeatureEngineering ───────────────────────────────────────
    py::class_<FeatureEngineering>(m, "FeatureEngineering")
        .def_static("compute_rsi",          &FeatureEngineering::compute_rsi,
                    py::arg("prices"), py::arg("period")=14)
        .def_static("compute_macd",         &FeatureEngineering::compute_macd,
                    py::arg("prices"), py::arg("fast")=12,
                    py::arg("slow")=26, py::arg("signal")=9)
        .def_static("compute_bollinger",    &FeatureEngineering::compute_bollinger,
                    py::arg("prices"), py::arg("period")=20)
        .def_static("compute_atr",          &FeatureEngineering::compute_atr,
                    py::arg("prices"), py::arg("period")=14)
        .def_static("compute_williams_r",   &FeatureEngineering::compute_williams_r,
                    py::arg("prices"), py::arg("period")=14)
        .def_static("compute_cci",          &FeatureEngineering::compute_cci,
                    py::arg("prices"), py::arg("period")=20)
        .def_static("compute_stochastic_k", &FeatureEngineering::compute_stochastic_k,
                    py::arg("prices"), py::arg("k")=14)
        .def_static("compute_log_returns",  &FeatureEngineering::compute_log_returns)
        .def_static("compute_realised_vol", &FeatureEngineering::compute_realised_vol,
                    py::arg("returns"), py::arg("window")=20)
        .def_static("build_feature_matrix", &FeatureEngineering::build_feature_matrix,
                    py::arg("prices"), py::arg("volumes")=std::vector<float>{});

    // ── DataNormalizer ───────────────────────────────────────────
    py::class_<DataNormalizer>(m, "DataNormalizer")
        .def(py::init<>())
        .def("fit",               &DataNormalizer::fit,
             py::arg("data"), py::arg("robust")=false)
        .def("transform",         &DataNormalizer::transform)
        .def("inverse_transform", &DataNormalizer::inverse_transform)
        .def("inverse_scalar",    &DataNormalizer::inverse_scalar)
        .def_readwrite("mean_",   &DataNormalizer::mean_)
        .def_readwrite("std_",    &DataNormalizer::std_)
        .def_readwrite("median_", &DataNormalizer::median_)
        .def_readwrite("iqr_",    &DataNormalizer::iqr_)
        .def_readwrite("fitted_", &DataNormalizer::fitted_);

    // ── BacktestEngine ───────────────────────────────────────────
    py::class_<BacktestEngine>(m, "BacktestEngine")
        .def(py::init<>())
        .def_static("run", &BacktestEngine::run,
             py::arg("prices"), py::arg("predictor"),
             py::arg("initial_capital")=10000.0f,
             py::arg("transaction_cost")=0.001f);

    // ── AutoHyperparamSearch ─────────────────────────────────────
    py::class_<AutoHyperparamSearch>(m, "AutoHyperparamSearch")
        .def(py::init<int>(), py::arg("n_trials")=20)
        .def("search", &AutoHyperparamSearch::search,
             py::arg("prices"), py::arg("n_trials")=20);

    // ── Utility functions ────────────────────────────────────────
    m.def("numa_available", []() -> bool {
#ifdef HAVE_NUMA
        return true;
#else
        return false;
#endif
    });

    m.def("numa_num_nodes", []() -> int {
#ifdef HAVE_NUMA
        return numa_num_configured_nodes();
#else
        return 1;
#endif
    });

    m.def("build_windows", [](const std::vector<float>& prices, int seq_len) {
        std::vector<std::pair<std::vector<float>,float>> windows;
        for (size_t i=0;i+seq_len<prices.size();++i) {
            std::vector<float> w(prices.begin()+i, prices.begin()+i+seq_len);
            windows.push_back({w, prices[i+seq_len]});
        }
        return windows;
    }, py::arg("prices"), py::arg("seq_len")=60);

    m.def("model_names", []() -> std::vector<std::string> {
        return {
            "LSTM", "Transformer", "TCN", "WaveNet", "N-BEATS",
            "Informer", "NHiTS", "TFT", "PatchTST", "TimesNet",
            "DLinear", "Crossformer"
        };
    });

    m.def("model_ids", []() -> std::vector<std::string> {
        return {
            "lstm", "transformer", "tcn", "wavenet", "nbeats",
            "informer", "nhits", "tft", "patchtst", "timesnet",
            "dlinear", "crossformer"
        };
    });

    m.def("version", []() -> std::string { return "3.0.0-12models"; });
}
