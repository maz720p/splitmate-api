# SplitMate API Services

Dua service FastAPI terpisah untuk prediksi pengeluaran dan rekomendasi keuangan.

---

## Struktur Folder

```
splitmate/
├── expense_api.py
├── genai_api.py
├── requirements.txt
├── run.bat
├── README.md
└── model_exports/
    ├── expense_predictor.keras
    ├── scalers.pkl
    └── model_config.json
```

---

## Cara Menjalankan (Windows)

Buka CMD di dalam folder splitmate, lalu jalankan:

```
run.bat
```

Script akan otomatis:
1. Cek Python tersedia
2. Install semua dependencies
3. Minta Anthropic API Key jika belum diset
4. Menjalankan kedua API di window CMD terpisah

Setelah jalan, buka browser ke:
- http://localhost:8000/docs untuk Expense Prediction API
- http://localhost:8001/docs untuk GenAI Insight API

---

## 1. expense_api.py

FastAPI untuk prediksi pengeluaran bulanan menggunakan model LSTM. Berjalan di port 8000.

Endpoints:
- GET /health - status model
- GET /model-info - metrik model (MAE, MAPE, R2)
- POST /predict - prediksi pengeluaran

Contoh request /predict:

```
POST http://localhost:8000/predict
Content-Type: application/json

{
  "transactions": [
    {"year_month": "2024-01", "amount": 1500000, "category": "food"},
    {"year_month": "2024-01", "amount": 300000, "category": "transport"}
  ],
  "n_months_ahead": 3
}
```

---

## 2. genai_api.py

FastAPI untuk rekomendasi keuangan menggunakan Anthropic Claude. Berjalan di port 8001.

Endpoints:
- GET /health - status key dan URL expense API
- POST /insight - pipeline penuh: prediksi LSTM + rekomendasi AI
- POST /recommend - hanya rekomendasi AI dari prediksi yang sudah ada

Contoh request /insight:

```
POST http://localhost:8001/insight
Content-Type: application/json

{
  "transactions": [...],
  "n_months_ahead": 3,
  "api_key": "sk-ant-..."
}
```

---

## Environment Variables

- ANTHROPIC_API_KEY - Anthropic API key (wajib untuk genai_api)
- EXPENSE_API_URL - URL expense prediction service, default http://localhost:8000
- MODEL_DIR - folder model artifacts, default model_exports

---

## Catatan

- Format year_month harus "YYYY-MM", contoh "2024-03"
- Data minimal yang dibutuhkan sesuai window_size di model_config.json
- Jalankan CMD sebagai Administrator jika pip install gagal karena permission
