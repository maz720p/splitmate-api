"""
expense_api.py  ─  SplitMate Expense Prediction API
====================================================
FastAPI service untuk prediksi pengeluaran bulanan menggunakan model LSTM
yang sudah di-train dan di-export ke model_exports/.

Struktur direktori yang dibutuhkan:
  model_exports/
    expense_predictor.keras
    scalers.pkl
    model_config.json

Jalankan:
  uvicorn expense_api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import pickle
import json
import os
import tensorflow as tf
from tensorflow import keras

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SplitMate Expense Prediction API",
    description="REST API untuk prediksi pengeluaran bulanan menggunakan LSTM",
    version="1.0.0"
)


# ── Custom classes (dibutuhkan untuk load model) ──────────────────────────────

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


# ── Load artefak model ────────────────────────────────────────────────────────

MODEL_DIR = os.environ.get("MODEL_DIR", "model_exports")

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


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TransactionItem(BaseModel):
    year_month      : str          # format: "YYYY-MM"
    amount          : float
    category        : str
    transaction_type: str = "expense"


class PredictRequest(BaseModel):
    transactions  : List[TransactionItem]
    n_months_ahead: Optional[int] = 1


class PredictionResult(BaseModel):
    bulan_ke    : int
    prediksi_rp : float
    prediksi_fmt: str


class PredictResponse(BaseModel):
    status        : str
    window_months : int
    predictions   : List[PredictionResult]
    mae_normalized: float


# ── Helper ────────────────────────────────────────────────────────────────────

def build_features(df_expense: pd.DataFrame) -> pd.DataFrame:
    """Membangun feature matrix dari data transaksi expense."""
    monthly = (
        df_expense.groupby("year_month")["amount"]
        .sum().reset_index()
        .rename(columns={"amount": "total_expense"})
        .sort_values("year_month").reset_index(drop=True)
    )
    monthly_cat = (
        df_expense.groupby(["year_month", "category"])["amount"]
        .sum().unstack(fill_value=0).reset_index()
        .sort_values("year_month").reset_index(drop=True)
    )
    feat = monthly.merge(monthly_cat, on="year_month", how="left")
    feat["month_num"] = feat["year_month"].apply(lambda x: int(x.split("-")[1]))
    feat["year_num"]  = feat["year_month"].apply(lambda x: int(x.split("-")[0]))
    feat["month_sin"] = np.sin(2 * np.pi * feat["month_num"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month_num"] / 12)
    feat["lag_1"]     = feat["total_expense"].shift(1)
    feat["lag_2"]     = feat["total_expense"].shift(2)
    feat["lag_3"]     = feat["total_expense"].shift(3)
    feat["rolling_3"] = feat["total_expense"].rolling(3).mean().shift(1)
    feat["rolling_6"] = feat["total_expense"].rolling(6).mean().shift(1)
    feat.dropna(inplace=True)
    feat.reset_index(drop=True, inplace=True)
    # pastikan semua kolom ada (isi 0 jika kategori tidak ada di data ini)
    for col in FEATURE_COLS:
        if col not in feat.columns:
            feat[col] = 0.0
    return feat


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service"  : "SplitMate Expense Prediction API",
        "version"  : "1.0.0",
        "endpoints": ["/predict", "/health", "/model-info", "/docs"]
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded is not None}


@app.get("/model-info")
def model_info():
    return {
        "model_type"  : model_config["model_type"],
        "window_size" : model_config["window_size"],
        "n_features"  : model_config["n_features"],
        "test_mae_norm": model_config["test_mae_norm"],
        "test_mae_rp" : model_config["test_mae_rp"],
        "test_mape_pct": model_config["test_mape_pct"],
        "test_r2"     : model_config["test_r2"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Prediksi total pengeluaran bulanan ke depan.

    - **transactions**: list transaksi historis (minimal WINDOW bulan)
    - **n_months_ahead**: berapa bulan ke depan yang ingin diprediksi (default 1)
    """
    df = pd.DataFrame([t.dict() for t in req.transactions])
    df = df[df["transaction_type"] == "expense"]

    if df.empty:
        raise HTTPException(status_code=400, detail="Tidak ada data expense dalam transaksi yang dikirim")

    feat = build_features(df)
    if len(feat) < WINDOW:
        raise HTTPException(
            status_code=400,
            detail=f"Data tidak cukup. Butuh minimal {WINDOW} bulan, tersedia {len(feat)} bulan."
        )

    X     = scaler_X.transform(feat[FEATURE_COLS].values)
    X_win = X[-WINDOW:].copy()
    preds = []

    for step in range(req.n_months_ahead):
        X_in = X_win[-WINDOW:].reshape(1, WINDOW, -1)
        y_sc = float(model_loaded.predict(X_in, verbose=0)[0, 0])
        y_rp = float(scaler_y.inverse_transform([[y_sc]])[0, 0])
        preds.append(PredictionResult(
            bulan_ke    =step + 1,
            prediksi_rp =y_rp,
            prediksi_fmt=f"Rp {y_rp:,.0f}"
        ))
        # update sliding window dengan prediksi sebagai input berikutnya
        new_row    = X_win[-1].copy()
        new_row[0] = y_sc
        X_win      = np.vstack([X_win[1:], new_row])

    return PredictResponse(
        status        ="success",
        window_months =WINDOW,
        predictions   =preds,
        mae_normalized=model_config["test_mae_norm"]
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
