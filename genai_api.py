"""
genai_api.py  -  SplitMate Generative AI (Financial Insight) API
=================================================================
FastAPI service untuk rekomendasi keuangan berbasis Gemini API.
Memanggil expense_api (/predict) lalu menghasilkan insight AI.

Jalankan:
  GEMINI_API_KEY=... uvicorn genai_api:app --host 0.0.0.0 --port 8001 --reload

Pastikan expense_api sudah berjalan di port 8000 (atau set EXPENSE_API_URL).
"""

import os
import httpx
import google.generativeai as genai
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
EXPENSE_API_URL = os.environ.get("EXPENSE_API_URL", "http://localhost:8000")

app = FastAPI(
    title="SplitMate GenAI Insight API",
    description="REST API rekomendasi keuangan menggunakan Gemini API",
    version="2.0.0"
)


# Pydantic schemas — harus identik dengan FeatureRow di expense_api.py (21 fitur model v4)

class FeatureRow(BaseModel):
    amount_idr            : float          # IDR mentah
    month                 : float
    is_weekend            : int
    year                  : float
    category_makanan      : int = 0
    category_tempat_tinggal: int = 0
    category_transportasi : int = 0
    category_tagihan      : int = 0
    category_hiburan      : int = 0
    category_kesehatan    : int = 0
    category_pendidikan   : int = 0
    category_tabungan     : int = 0
    category_lainnya      : int = 0
    category_investasi    : int = 0
    day_of_week_Friday    : int = 0
    day_of_week_Monday    : int = 0
    day_of_week_Saturday  : int = 0
    day_of_week_Sunday    : int = 0
    day_of_week_Thursday  : int = 0
    day_of_week_Tuesday   : int = 0
    day_of_week_Wednesday : int = 0


class PredictionResult(BaseModel):
    step          : int
    prediksi_norm : float
    prediksi_idr  : float
    prediksi_fmt  : str


class InsightRequest(BaseModel):
    rows          : List[FeatureRow]
    n_steps_ahead : Optional[int] = 3
    api_key       : Optional[str] = None


class InsightSummary(BaseModel):
    n_rows_input   : int
    avg_amount_idr : float
    top_category   : str


class InsightResponse(BaseModel):
    status         : str
    predictions    : List[PredictionResult]
    recommendation : str
    summary        : InsightSummary


class RecommendRequest(BaseModel):
    rows        : List[FeatureRow]
    predictions : List[PredictionResult]
    api_key     : Optional[str] = None


class RecommendResponse(BaseModel):
    status        : str
    recommendation: str


# Prompt builder

CATEGORY_COLS = [
    "category_makanan", "category_tempat_tinggal", "category_transportasi",
    "category_tagihan", "category_hiburan", "category_kesehatan",
    "category_pendidikan", "category_tabungan", "category_lainnya", "category_investasi"
]


def _build_prompt(rows: List[FeatureRow], predictions: List[PredictionResult]) -> str:
    cat_totals: Dict[str, int] = {c: 0 for c in CATEGORY_COLS}
    for row in rows:
        for c in CATEGORY_COLS:
            cat_totals[c] += getattr(row, c, 0)

    top_category = max(cat_totals, key=lambda k: cat_totals[k]).replace("category_", "")
    avg_amount   = float(np.mean([r.amount_idr for r in rows]))

    pred_lines = [
        f"  Langkah +{p.step}: {p.prediksi_fmt} (normalized: {p.prediksi_norm:.4f})"
        for p in predictions
    ]

    return f"""Kamu adalah asisten keuangan cerdas untuk aplikasi SplitMate, \
aplikasi manajemen patungan untuk Gen Z Indonesia.

Ringkasan data input pengguna:
- Jumlah transaksi terakhir : {len(rows)}
- Rata-rata pengeluaran     : Rp {avg_amount:,.0f}
- Kategori terbanyak        : {top_category}

Prediksi dari model BiLSTM + Attention:
{chr(10).join(pred_lines)}

Berdasarkan data di atas, berikan:
1. Analisis singkat pola pengeluaran (2-3 kalimat)
2. 3 rekomendasi konkret untuk mengelola keuangan ke depan
3. Tips khusus untuk kategori terbanyak: {top_category}

Gunakan Bahasa Indonesia yang ramah dan relevan untuk Gen Z.
Jawaban maksimal 250 kata."""


def get_financial_recommendation(
    rows: List[FeatureRow],
    predictions: List[PredictionResult],
    api_key: str = None
) -> str:
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError(
            "GEMINI_API_KEY belum diset. "
            "Set environment variable atau kirim api_key dalam request."
        )

    genai.configure(api_key=key)
    model    = genai.GenerativeModel("gemini-2.5-flash")
    prompt   = _build_prompt(rows, predictions)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.7, "max_output_tokens": 512}
    )
    return response.text


# Endpoints

@app.get("/")
def root():
    return {
        "service"  : "SplitMate GenAI Insight API",
        "version"  : "2.0.0",
        "endpoints": ["/insight", "/recommend", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status"         : "ok",
        "gemini_key"     : "set" if GEMINI_API_KEY else "NOT SET",
        "expense_api_url": EXPENSE_API_URL
    }


@app.post("/insight", response_model=InsightResponse)
def insight(req: InsightRequest):
    """
    Pipeline lengkap: kirim baris fitur -> prediksi LSTM -> rekomendasi AI Gemini.
    amount_idr dalam IDR mentah (bukan dibagi juta).
    """
    payload = {
        "rows"         : [r.model_dump() for r in req.rows],
        "n_steps_ahead": req.n_steps_ahead
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

    predictions = [PredictionResult(**p) for p in pred_data["predictions"]]

    cat_totals  = {c: sum(getattr(r, c, 0) for r in req.rows) for c in CATEGORY_COLS}
    top_cat     = max(cat_totals, key=lambda k: cat_totals[k]).replace("category_", "")
    avg_amount  = float(np.mean([r.amount_idr for r in req.rows]))

    try:
        recommendation = get_financial_recommendation(req.rows, predictions, req.api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=401, detail="Gemini API key tidak valid atau quota habis")

    return InsightResponse(
        status        ="success",
        predictions   =predictions,
        recommendation=recommendation,
        summary       =InsightSummary(
            n_rows_input   =len(req.rows),
            avg_amount_idr =avg_amount,
            top_category   =top_cat
        )
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """Generate rekomendasi AI dari data dan prediksi yang sudah ada."""
    try:
        recommendation = get_financial_recommendation(req.rows, req.predictions, req.api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=401, detail="Gemini API key tidak valid atau quota habis")

    return RecommendResponse(status="success", recommendation=recommendation)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
