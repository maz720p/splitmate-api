"""
genai_api.py  ─  SplitMate Generative AI (Financial Insight) API
=================================================================
FastAPI service terpisah untuk rekomendasi keuangan berbasis Anthropic Claude.
Service ini memanggil expense_api (/predict) lalu menghasilkan insight AI.

Jalankan:
  ANTHROPIC_API_KEY=sk-ant-... uvicorn genai_api:app --host 0.0.0.0 --port 8001 --reload

Atau set di .env / environment variable sebelum menjalankan.

Catatan: pastikan expense_api sudah berjalan di port 8000 (EXPENSE_API_URL).
"""

import os
import httpx
import anthropic
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
EXPENSE_API_URL   = os.environ.get("EXPENSE_API_URL", "http://localhost:8000")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SplitMate GenAI Insight API",
    description="REST API rekomendasi keuangan menggunakan Anthropic Claude",
    version="1.0.0"
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TransactionItem(BaseModel):
    year_month      : str   # format: "YYYY-MM"
    amount          : float
    category        : str
    transaction_type: str = "expense"


class InsightRequest(BaseModel):
    transactions   : List[TransactionItem]
    n_months_ahead : Optional[int] = 3
    api_key        : Optional[str] = None   # override env var jika perlu


class PredictionResult(BaseModel):
    bulan_ke    : int
    prediksi_rp : float
    prediksi_fmt: str


class InsightSummary(BaseModel):
    total_expense_rp : float
    avg_monthly_rp   : float
    top_category     : str
    months_analyzed  : int


class InsightResponse(BaseModel):
    status         : str
    predictions    : List[PredictionResult]
    recommendation : str
    summary        : InsightSummary


class RecommendRequest(BaseModel):
    """Request langsung tanpa memanggil LSTM (pakai data & prediksi manual)."""
    transactions   : List[TransactionItem]
    predictions    : List[PredictionResult]
    api_key        : Optional[str] = None


class RecommendResponse(BaseModel):
    status        : str
    recommendation: str


# ── Core AI function ──────────────────────────────────────────────────────────

def _build_prompt(df_expense: pd.DataFrame, predictions: list) -> str:
    """Menyusun prompt untuk Claude dari data expense + prediksi LSTM."""
    monthly_summary = (
        df_expense.groupby("year_month")["amount"]
        .sum().reset_index()
        .rename(columns={"amount": "total"})
        .tail(6)
    )
    category_summary = (
        df_expense.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    pred_lines = [
        f"  Bulan +{p['bulan_ke']}: {p['prediksi_fmt']}"
        for p in predictions
    ]

    prompt = f"""Kamu adalah asisten keuangan cerdas untuk aplikasi SplitMate, \
aplikasi manajemen patungan untuk Gen Z.

Data pengeluaran pengguna 6 bulan terakhir:
{monthly_summary.to_string(index=False)}

5 kategori pengeluaran terbesar:
{category_summary.to_string()}

Prediksi pengeluaran dari model LSTM:
{chr(10).join(pred_lines)}

Berdasarkan data di atas, berikan:
1. Analisis singkat pola pengeluaran (2-3 kalimat)
2. 3 rekomendasi konkret untuk menghemat pengeluaran bulan depan
3. Satu tips khusus untuk kategori pengeluaran terbesar

Gunakan Bahasa Indonesia yang ramah dan relevan untuk Gen Z.
Jawaban maksimal 250 kata."""

    return prompt


def get_financial_recommendation(
    df_expense: pd.DataFrame,
    predictions: list,
    api_key: str = None
) -> str:
    """
    Kirim prompt ke Anthropic Claude dan kembalikan rekomendasi keuangan.

    Args:
        df_expense  : DataFrame dengan kolom year_month, amount, category
        predictions : list dict dari /predict endpoint
        api_key     : Anthropic API key (opsional, fallback ke env var)

    Returns:
        str: teks rekomendasi dalam Bahasa Indonesia
    """
    key = api_key or ANTHROPIC_API_KEY
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY belum diset. "
            "Set environment variable atau kirim api_key dalam request."
        )

    client = anthropic.Anthropic(api_key=key)
    prompt = _build_prompt(df_expense, predictions)

    response = client.messages.create(
        model     ="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages  =[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service"  : "SplitMate GenAI Insight API",
        "version"  : "1.0.0",
        "endpoints": ["/insight", "/recommend", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status"         : "ok",
        "anthropic_key"  : "set" if ANTHROPIC_API_KEY else "NOT SET – kirim api_key di request",
        "expense_api_url": EXPENSE_API_URL
    }


@app.post("/insight", response_model=InsightResponse)
def insight(req: InsightRequest):
    """
    Pipeline lengkap: kirim transaksi → prediksi LSTM → rekomendasi AI.

    - Memanggil expense_api (/predict) untuk mendapatkan prediksi
    - Kemudian menghasilkan rekomendasi keuangan via Claude
    """
    # 1. Panggil expense_api /predict
    payload = {
        "transactions"  : [t.dict() for t in req.transactions],
        "n_months_ahead": req.n_months_ahead
    }
    try:
        with httpx.Client(timeout=30) as client:
            r = client.post(f"{EXPENSE_API_URL}/predict", json=payload)
            r.raise_for_status()
            pred_data = r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Expense API error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Expense API tidak dapat dijangkau: {str(e)}")

    predictions = pred_data["predictions"]

    # 2. Buat summary data
    df_expense = pd.DataFrame([t.dict() for t in req.transactions])
    df_expense = df_expense[df_expense["transaction_type"] == "expense"]

    if df_expense.empty:
        raise HTTPException(status_code=400, detail="Tidak ada data expense")

    total_expense = float(df_expense["amount"].sum())
    avg_monthly   = float(df_expense.groupby("year_month")["amount"].sum().mean())
    top_category  = df_expense.groupby("category")["amount"].sum().idxmax()
    months_count  = df_expense["year_month"].nunique()

    # 3. Generate rekomendasi AI
    try:
        recommendation = get_financial_recommendation(df_expense, predictions, req.api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Anthropic API key tidak valid")

    return InsightResponse(
        status      ="success",
        predictions =[PredictionResult(**p) for p in predictions],
        recommendation=recommendation,
        summary     =InsightSummary(
            total_expense_rp=total_expense,
            avg_monthly_rp  =avg_monthly,
            top_category    =top_category,
            months_analyzed =months_count
        )
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Hanya generate rekomendasi AI dari data & prediksi yang sudah ada.
    Gunakan ini jika prediksi sudah didapat dari expense_api secara terpisah.
    """
    df_expense = pd.DataFrame([t.dict() for t in req.transactions])
    df_expense = df_expense[df_expense["transaction_type"] == "expense"]

    if df_expense.empty:
        raise HTTPException(status_code=400, detail="Tidak ada data expense")

    try:
        recommendation = get_financial_recommendation(
            df_expense,
            [p.dict() for p in req.predictions],
            req.api_key
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Anthropic API key tidak valid")

    return RecommendResponse(status="success", recommendation=recommendation)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
