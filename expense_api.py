"""
expense_api.py  -  SplitMate Expense Prediction API
====================================================
FastAPI service untuk prediksi pengeluaran menggunakan model LSTM
yang sudah di-train dan di-export ke model_exports/.

Struktur direktori yang dibutuhkan:
  model_exports/
    expense_predictor.keras
    scalers.pkl
    model_config.json
  logs/                          <-- TensorBoard training logs
    20260511-075444/
      train/
      validation/
    gradient_tape_20260511-075701/
      train/
      validation/

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

# App

app = FastAPI(
    title="SplitMate Expense Prediction API",
    description="REST API untuk prediksi pengeluaran menggunakan LSTM",
    version="1.1.0"
)


# Custom classes (dibutuhkan untuk load model)

class AttentionLayer(keras.layers.Layer):
    """Self-attention Bahdanau-style pada output LSTM."""

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(name="W", shape=(input_shape[-1], self.units), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(self.units, 1), initializer="glorot_uniform")

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b)
        attn  = tf.nn.softmax(tf.squeeze(tf.tensordot(score, self.v, axes=[[2], [0]]), axis=-1), axis=1)
        ctx   = tf.reduce_sum(inputs * tf.expand_dims(attn, -1), axis=1)
        return ctx, attn

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


class WeightedHuberLoss(keras.losses.Loss):
    """Huber loss asimetris: penalti lebih besar jika over-predict."""

    def __init__(self, delta: float = 0.1, over_penalty: float = 1.5,
                 under_penalty: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta         = delta
        self.over_penalty  = over_penalty
        self.under_penalty = under_penalty

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        err    = y_pred - y_true
        huber  = tf.where(tf.abs(err) <= self.delta,
                          0.5 * tf.square(err),
                          self.delta * (tf.abs(err) - 0.5 * self.delta))
        w = tf.where(err > 0, self.over_penalty, self.under_penalty)
        return tf.reduce_mean(w * huber)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"delta": self.delta, "over_penalty": self.over_penalty,
                    "under_penalty": self.under_penalty})
        return cfg


# Load artefak model

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

WINDOW       = model_config["window_size"]
FEATURE_COLS = model_config["feature_cols"]
N_FEATURES   = model_config["n_features"]


# Pydantic schemas

class FeatureRow(BaseModel):
    amount: float
    month: float
    is_weekend: int
    year: float
    transaction_type_income: int
    category_freelance: int = 0
    category_gaji: int = 0
    category_hiburan: int = 0
    category_investasi: int = 0
    category_kesehatan: int = 0
    category_lainnya: int = 0
    category_makanan: int = 0
    category_pendidikan: int = 0
    category_tabungan: int = 0
    category_tagihan: int = 0
    category_tempat_tinggal: int = 0
    category_transportasi: int = 0
    payment_mode_ewallet: int = 0
    payment_mode_kartu: int = 0
    payment_mode_qris: int = 0
    payment_mode_tunai: int = 0
    location_bandung: int = 0
    location_denpasar: int = 0
    location_jakarta: int = 0
    location_makassar: int = 0
    location_medan: int = 0
    location_palembang: int = 0
    location_semarang: int = 0
    location_surabaya: int = 0
    location_unknown: int = 0
    location_yogyakarta: int = 0
    day_of_week_Monday: int = 0
    day_of_week_Saturday: int = 0
    day_of_week_Sunday: int = 0
    day_of_week_Thursday: int = 0
    day_of_week_Tuesday: int = 0
    day_of_week_Wednesday: int = 0


class PredictRequest(BaseModel):
    rows: List[FeatureRow]
    n_steps_ahead: Optional[int] = 1


class PredictionResult(BaseModel):
    step: int
    prediksi_norm: float
    prediksi_idr: float
    prediksi_fmt: str


class PredictResponse(BaseModel):
    status: str
    window_size: int
    predictions: List[PredictionResult]
    mae_normalized: float


# Endpoints

@app.get("/")
def root():
    return {
        "service"  : "SplitMate Expense Prediction API",
        "version"  : "1.1.0",
        "endpoints": ["/predict", "/health", "/model-info", "/training-logs", "/training-metrics", "/training-metrics/{run_name}", "/docs"]
    }


@app.get("/health")
def health():
    logs_present = os.path.isdir(LOGS_DIR) and bool(os.listdir(LOGS_DIR))
    return {
        "status"       : "ok",
        "model_loaded" : model_loaded is not None,
        "logs_present" : logs_present,
        "logs_dir"     : LOGS_DIR,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type"    : model_config["model_type"],
        "window_size"   : model_config["window_size"],
        "n_features"    : model_config["n_features"],
        "test_mae_norm" : model_config["test_mae_norm"],
        "test_accuracy" : model_config["test_accuracy"],
        "accuracy_def"  : model_config["accuracy_def"],
    }


@app.get("/training-logs")
def training_logs():
    """
    Daftar TensorBoard log runs yang tersedia di folder logs/.
    Untuk membuka TensorBoard: tensorboard --logdir logs/
    """
    if not os.path.isdir(LOGS_DIR):
        raise HTTPException(
            status_code=404,
            detail=f"Folder logs tidak ditemukan: {LOGS_DIR}."
        )

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
    """
    Ringkasan metrik semua run (tanpa series lengkap).
    Membaca event files via tensorboard_utils.
    """
    runs = tensorboard_utils.get_all_runs_summary()
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=f"Tidak ada run ditemukan di folder logs: {LOGS_DIR}"
        )
    return {"status": "ok", "runs": runs}


@app.get("/training-metrics/{run_name}")
def training_metrics_run(run_name: str):
    """
    Metrik lengkap (termasuk series per step) untuk satu run tertentu.
    Contoh: GET /training-metrics/20260511-075444
    """
    data = tensorboard_utils.get_run_metrics(run_name)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_name}' tidak ditemukan di {LOGS_DIR}"
        )
    return {"status": "ok", **data}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Prediksi amount_idr step berikutnya.

    - **rows**: list baris fitur sudah ternormalisasi (minimal WINDOW baris)
    - **n_steps_ahead**: berapa langkah ke depan yang ingin diprediksi (default 1)
    """
    if len(req.rows) < WINDOW:
        raise HTTPException(
            status_code=400,
            detail=f"Butuh minimal {WINDOW} baris, tersedia {len(req.rows)}."
        )

    X_raw = np.array([
        [getattr(row, col, 0.0) for col in FEATURE_COLS]
        for row in req.rows
    ], dtype=np.float32)

    X_win = X_raw[-WINDOW:].copy()
    preds = []

    for step in range(req.n_steps_ahead):
        X_in   = X_win[-WINDOW:].reshape(1, WINDOW, N_FEATURES)
        y_norm = float(model_loaded.predict(X_in, verbose=0)[0, 0])
        y_idr  = float(scaler_y.inverse_transform([[y_norm]])[0, 0])

        preds.append(PredictionResult(
            step          =step + 1,
            prediksi_norm =y_norm,
            prediksi_idr  =y_idr,
            prediksi_fmt  =f"Rp {y_idr:,.0f}"
        ))

        new_row = X_win[-1].copy()
        if "amount_idr" in FEATURE_COLS:
            new_row[FEATURE_COLS.index("amount_idr")] = y_norm
        X_win = np.vstack([X_win[1:], new_row])

    return PredictResponse(
        status        ="success",
        window_size   =WINDOW,
        predictions   =preds,
        mae_normalized=model_config["test_mae_norm"]
    )


# Entry point

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
