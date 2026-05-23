"""
expense_api.py  -  SplitMate Expense Prediction API
====================================================
FastAPI service untuk prediksi pengeluaran menggunakan model BiLSTM + Attention
yang sudah di-train dan di-export ke model_exports/.

Model v4: 21 fitur input, target log-transformed, MAE 0.0158, Accuracy 97.9%

Struktur direktori yang dibutuhkan:
  model_exports/
    expense_predictor.keras
    scalers.pkl
    model_config.json
  logs/
    20260520-221031/
      train/  validation/  gt_train/  gt_val/

Jalankan:
  uvicorn expense_api:app --host 0.0.0.0 --port 8000 --reload

Untuk TensorBoard:
  tensorboard --logdir logs/
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pickle
import json
import os
import glob
import tensorflow as tf
from tensorflow import keras
import tensorboard_utils

app = FastAPI(
    title="SplitMate Expense Prediction API",
    description="REST API prediksi pengeluaran menggunakan BiLSTM + Attention (model v4)",
    version="2.0.0"
)


# Custom classes — wajib sama persis dengan yang dipakai saat training

class AttentionLayer(keras.layers.Layer):
    """Bahdanau-style Additive Attention pada output BiLSTM."""

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        dim = input_shape[-1]
        self.W = self.add_weight(shape=(dim, self.units), name="W", initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(self.units,),     name="b", initializer="zeros",          trainable=True)
        self.v = self.add_weight(shape=(self.units, 1),   name="v", initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b)
        energy = tf.squeeze(tf.tensordot(score, self.v, axes=[[2], [0]]), axis=-1)
        attn   = tf.nn.softmax(energy, axis=1)
        ctx    = tf.reduce_sum(inputs * tf.expand_dims(attn, -1), axis=1)
        return ctx, attn

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


class WeightedHuberLoss(keras.losses.Loss):
    """Asymmetric Huber Loss: penalti lebih besar untuk over-prediction."""

    def __init__(self, delta: float = 0.02, over_penalty: float = 1.5,
                 under_penalty: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta         = delta
        self.over_penalty  = over_penalty
        self.under_penalty = under_penalty

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        err    = y_pred - y_true
        abs_e  = tf.abs(err)
        huber  = tf.where(abs_e <= self.delta,
                          0.5 * tf.square(err),
                          self.delta * (abs_e - 0.5 * self.delta))
        w = tf.where(err > 0, self.over_penalty, self.under_penalty)
        return tf.reduce_mean(w * huber)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"delta": self.delta, "over_penalty": self.over_penalty,
                    "under_penalty": self.under_penalty})
        return cfg


# Load artifacts

MODEL_DIR = os.environ.get("MODEL_DIR", "model_exports")
LOGS_DIR  = os.environ.get("LOGS_DIR",  "logs")

model_loaded = keras.models.load_model(
    os.path.join(MODEL_DIR, "expense_predictor.keras"),
    custom_objects={"AttentionLayer": AttentionLayer, "WeightedHuberLoss": WeightedHuberLoss}
)

with open(os.path.join(MODEL_DIR, "scalers.pkl"), "rb") as f:
    scalers = pickle.load(f)
scaler_X = scalers["scaler_X"]
scaler_y = scalers["scaler_y"]

with open(os.path.join(MODEL_DIR, "model_config.json")) as f:
    model_config = json.load(f)

WINDOW       = model_config["window_size"]     # 10
FEATURE_COLS = model_config["feature_cols"]    # 21 kolom (tanpa payment_mode & location)
N_FEATURES   = model_config["n_features"]      # 21


# Pydantic schemas — sesuai 21 fitur model v4
# Tidak ada: payment_mode, location, transaction_type_income, category_freelance, category_gaji
# day_of_week_Friday sekarang ada (bukan reference category)

class FeatureRow(BaseModel):
    amount_idr           : float          # IDR mentah, backend akan normalize via scaler_X
    month                : float          # 1-12
    is_weekend           : int            # 0 atau 1
    year                 : float          # 2022, 2023, 2024, 2025, 2026
    category_makanan     : int = 0
    category_tempat_tinggal: int = 0
    category_transportasi: int = 0
    category_tagihan     : int = 0
    category_hiburan     : int = 0
    category_kesehatan   : int = 0
    category_pendidikan  : int = 0
    category_tabungan    : int = 0
    category_lainnya     : int = 0
    category_investasi   : int = 0
    day_of_week_Friday   : int = 0        # Ada sebagai kolom eksplisit (bukan reference category)
    day_of_week_Monday   : int = 0
    day_of_week_Saturday : int = 0
    day_of_week_Sunday   : int = 0
    day_of_week_Thursday : int = 0
    day_of_week_Tuesday  : int = 0
    day_of_week_Wednesday: int = 0


class PredictRequest(BaseModel):
    rows          : List[FeatureRow]
    n_steps_ahead : Optional[int] = 1


class PredictionResult(BaseModel):
    step          : int
    prediksi_norm : float
    prediksi_idr  : float
    prediksi_fmt  : str


class PredictResponse(BaseModel):
    status        : str
    window_size   : int
    n_features    : int
    predictions   : List[PredictionResult]
    mae_normalized: float
    accuracy_pct  : float
    inference_note: str


# Endpoints

@app.get("/")
def root():
    return {
        "service"  : "SplitMate Expense Prediction API",
        "version"  : "2.0.0",
        "model"    : "BiLSTM + Attention v4 (21 fitur, MAE 0.0158, Accuracy 97.9%)",
        "endpoints": ["/predict", "/health", "/model-info", "/training-logs",
                      "/training-metrics", "/training-metrics/{run_name}", "/docs"]
    }


@app.get("/health")
def health():
    logs_present = os.path.isdir(LOGS_DIR) and bool(os.listdir(LOGS_DIR))
    return {
        "status"       : "ok",
        "model_loaded" : model_loaded is not None,
        "n_features"   : N_FEATURES,
        "window_size"  : WINDOW,
        "target_mae_met": model_config.get("target_mae_met", False),
        "target_acc_met": model_config.get("target_acc_met", False),
        "logs_present" : logs_present,
        "logs_dir"     : LOGS_DIR,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type"      : model_config["model_type"],
        "window_size"     : model_config["window_size"],
        "n_features"      : model_config["n_features"],
        "feature_cols"    : model_config["feature_cols"],
        "target_col"      : model_config["target_col"],
        "target_transform": model_config["target_transform"],
        "amount_unit"     : model_config["amount_unit"],
        "test_mae_norm"   : model_config["test_mae_norm"],
        "test_rmse_norm"  : model_config["test_rmse_norm"],
        "test_r2"         : model_config["test_r2"],
        "test_mape_pct"   : model_config["test_mape_pct"],
        "test_accuracy"   : model_config["test_accuracy"],
        "accuracy_def"    : model_config["accuracy_def"],
        "target_mae_met"  : model_config["target_mae_met"],
        "target_acc_met"  : model_config["target_acc_met"],
        "inference_note"  : model_config["inference_note"],
        "missing_features": model_config["missing_features"],
        "day_of_week_note": model_config["day_of_week_note"],
    }


@app.get("/training-logs")
def training_logs():
    """
    Daftar TensorBoard log runs yang tersedia di folder logs/.
    Untuk membuka TensorBoard: tensorboard --logdir logs/
    """
    if not os.path.isdir(LOGS_DIR):
        raise HTTPException(status_code=404, detail=f"Folder logs tidak ditemukan: {LOGS_DIR}")

    runs = []
    for entry in sorted(os.listdir(LOGS_DIR)):
        run_path = os.path.join(LOGS_DIR, entry)
        if not os.path.isdir(run_path):
            continue
        splits      = sorted(os.listdir(run_path))
        event_files = sorted(glob.glob(os.path.join(run_path, "**", "events.out.tfevents.*"), recursive=True))
        runs.append({
            "run_name"   : entry,
            "splits"     : splits,
            "event_files": [os.path.relpath(f, LOGS_DIR) for f in event_files],
        })

    return {
        "logs_dir"           : os.path.abspath(LOGS_DIR),
        "runs"               : runs,
        "tensorboard_command": f"tensorboard --logdir {LOGS_DIR}",
    }


@app.get("/training-metrics")
def training_metrics_all():
    """Ringkasan metrik semua run (tanpa series lengkap)."""
    runs = tensorboard_utils.get_all_runs_summary()
    if not runs:
        raise HTTPException(status_code=404, detail=f"Tidak ada run di: {LOGS_DIR}")
    return {"status": "ok", "runs": runs}


@app.get("/training-metrics/{run_name}")
def training_metrics_run(run_name: str):
    """
    Metrik lengkap untuk satu run.
    Contoh: GET /training-metrics/20260520-221031
    """
    data = tensorboard_utils.get_run_metrics(run_name)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' tidak ditemukan di {LOGS_DIR}")
    return {"status": "ok", **data}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Prediksi nominal transaksi berikutnya.

    Pipeline inference:
    1. Terima rows dalam IDR mentah (amount_idr bukan dibagi juta)
    2. Normalisasi via scaler_X.transform()
    3. Prediksi model (output log-normalized)
    4. Inverse: scaler_y.inverse_transform() -> np.expm1() -> IDR

    Cold start: jika rows < 10, zero-padding otomatis ditambahkan di depan.
    """
    rows = req.rows

    # Zero-padding untuk cold start (user < 10 transaksi)
    if len(rows) < WINDOW:
        X_raw = np.array([
            [getattr(row, col, 0.0) for col in FEATURE_COLS]
            for row in rows
        ], dtype=np.float32)
        pad   = np.zeros((WINDOW - len(X_raw), N_FEATURES), dtype=np.float32)
        X_raw = np.vstack([pad, X_raw])
    else:
        X_raw = np.array([
            [getattr(row, col, 0.0) for col in FEATURE_COLS]
            for row in rows
        ], dtype=np.float32)

    # Normalisasi input via scaler_X
    X_norm = scaler_X.transform(X_raw)
    X_win  = X_norm[-WINDOW:].copy()

    preds = []
    for step in range(req.n_steps_ahead):
        X_in      = X_win[-WINDOW:].reshape(1, WINDOW, N_FEATURES)
        y_norm    = float(model_loaded.predict(X_in, verbose=0)[0, 0])

        # Inverse transform: log-norm -> log -> IDR
        y_log = float(scaler_y.inverse_transform([[y_norm]])[0, 0])
        y_idr = float(np.expm1(y_log))

        preds.append(PredictionResult(
            step          =step + 1,
            prediksi_norm =round(y_norm, 6),
            prediksi_idr  =round(y_idr),
            prediksi_fmt  =f"Rp {y_idr:,.0f}"
        ))

        # Auto-regressive: update window dengan prediksi terakhir
        new_row = X_win[-1].copy()
        if "amount_idr" in FEATURE_COLS:
            new_row[FEATURE_COLS.index("amount_idr")] = y_norm
        X_win = np.vstack([X_win[1:], new_row])

    return PredictResponse(
        status        ="success",
        window_size   =WINDOW,
        n_features    =N_FEATURES,
        predictions   =preds,
        mae_normalized=model_config["test_mae_norm"],
        accuracy_pct  =model_config["test_accuracy"],
        inference_note=model_config["inference_note"]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
