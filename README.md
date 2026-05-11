# SplitMate API Services

Dua service FastAPI terpisah untuk prediksi pengeluaran dan rekomendasi keuangan.

---

## Struktur Folder

```text
splitmate/
├── expense_api.py
├── genai_api.py
├── tensorboard_utils.py          <-- utility baca TensorBoard event files
├── requirements.txt
├── README.md
├── model_exports/
│   ├── expense_predictor.keras
│   ├── scalers.pkl
│   └── model_config.json
└── logs/                         <-- TensorBoard training logs
    ├── 20260511-075444/
    │   ├── train/
    │   └── validation/
    └── gradient_tape_20260511-075701/
        ├── train/
        └── validation/
```

---

## Cara Menjalankan

Buka terminal di dalam folder `splitmate`, lalu:

```bash
pip install -r requirements.txt

# Terminal 1
uvicorn expense_api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2
GEMINI_API_KEY=AIza... uvicorn genai_api:app --host 0.0.0.0 --port 8001 --reload
```

Setelah berjalan, buka browser ke:
- http://localhost:8000/docs untuk Expense Prediction API
- http://localhost:8001/docs untuk GenAI Insight API

---

## TensorBoard

Folder `logs/` berisi training logs dari dua sesi training model LSTM:

| Run | Keterangan |
|---|---|
| `20260511-075444` | Sesi training utama (Keras fit callback) |
| `gradient_tape_20260511-075701` | Sesi training dengan custom `GradientTape` loop |

Masing-masing run memiliki dua split: `train` dan `validation`.

### Membuka TensorBoard (visual)

```bash
tensorboard --logdir logs/
```

Lalu buka http://localhost:6006

### Membaca metrik via API

`tensorboard_utils.py` membaca isi event files secara programatik dan dipakai oleh dua endpoint berikut:

| Endpoint | Keterangan |
|---|---|
| `GET /training-logs` | Daftar runs + path event files |
| `GET /training-metrics` | Ringkasan metrik semua run (tanpa series) |
| `GET /training-metrics/{run_name}` | Metrik lengkap + series per step untuk satu run |

Contoh response `GET /training-metrics/20260511-075444`:

```json
{
  "status": "ok",
  "run_name": "20260511-075444",
  "splits": ["train", "validation"],
  "summary": {
    "train": {
      "loss": { "n_steps": 50, "first": 0.12, "last": 0.004, "min": 0.004, "max": 0.12 }
    },
    "validation": {
      "loss": { "n_steps": 50, "first": 0.11, "last": 0.005, "min": 0.005, "max": 0.11 }
    }
  },
  "series": {
    "train": {
      "loss": [{ "step": 1, "value": 0.12 }, { "step": 2, "value": 0.09 }]
    }
  }
}
```

### Menggunakan `tensorboard_utils.py` secara langsung

```python
import tensorboard_utils

# Daftar semua run
runs = tensorboard_utils.list_runs()

# Ringkasan semua run
summary = tensorboard_utils.get_all_runs_summary()

# Metrik lengkap satu run
metrics = tensorboard_utils.get_run_metrics("20260511-075444")
```

### Mengganti lokasi folder logs

Set environment variable `LOGS_DIR` (berlaku untuk `expense_api.py` dan `tensorboard_utils.py`):

```bash
LOGS_DIR=/path/lain uvicorn expense_api:app --port 8000 --reload
```

---

## 1. expense_api.py

FastAPI untuk prediksi pengeluaran menggunakan model LSTM BiLSTM + Attention. Berjalan di port 8000.

Endpoints:
- `GET /health` - status model dan ketersediaan logs
- `GET /model-info` - metrik model (MAE, akurasi, window size)
- `GET /training-logs` - daftar TensorBoard log runs
- `GET /training-metrics` - ringkasan metrik semua run via `tensorboard_utils`
- `GET /training-metrics/{run_name}` - metrik lengkap satu run
- `POST /predict` - prediksi pengeluaran

Contoh request `/predict`:

```json
POST http://localhost:8000/predict
Content-Type: application/json

{
  "rows": [
    {
      "amount": -0.05,
      "month": -1.59,
      "is_weekend": 0,
      "year": -1.36,
      "transaction_type_income": 0,
      "category_makanan": 1,
      "payment_mode_ewallet": 1,
      "location_jakarta": 1,
      "day_of_week_Monday": 1
    }
  ],
  "n_steps_ahead": 3
}
```

Contoh response:

```json
{
  "status": "success",
  "window_size": 10,
  "predictions": [
    { "step": 1, "prediksi_norm": 0.032, "prediksi_idr": 850000.0, "prediksi_fmt": "Rp 850,000" }
  ],
  "mae_normalized": 0.006
}
```

---

## 2. genai_api.py

FastAPI untuk rekomendasi keuangan menggunakan Google Gemini. Berjalan di port 8001.

Endpoints:
- `GET /health` - status key dan URL expense API
- `POST /insight` - pipeline penuh: kirim rows → prediksi LSTM → rekomendasi AI
- `POST /recommend` - hanya rekomendasi AI dari prediksi yang sudah ada

Contoh request `/insight`:

```json
POST http://localhost:8001/insight
Content-Type: application/json

{
  "rows": [ ... ],
  "n_steps_ahead": 3,
  "api_key": "AIza..."
}
```

---

## 3. tensorboard_utils.py

Utility module untuk membaca TensorBoard event files secara programatik. Dipakai oleh `expense_api.py`.

| Fungsi | Keterangan |
|---|---|
| `list_runs()` | Daftar nama run di `LOGS_DIR` |
| `get_run_metrics(run_name)` | Metrik lengkap + series satu run |
| `get_all_runs_summary()` | Ringkasan semua run (tanpa series) |

---

## Daftar Field FeatureRow

| Field | Tipe | Keterangan |
|---|---|---|
| `amount` | float | Jumlah transaksi (normalized) |
| `month` | float | Bulan (normalized) |
| `is_weekend` | int | 1 jika weekend, 0 jika tidak |
| `year` | float | Tahun (normalized) |
| `transaction_type_income` | int | 1 jika income, 0 jika expense |
| `category_*` | int | One-hot: freelance, gaji, hiburan, investasi, kesehatan, lainnya, makanan, pendidikan, tabungan, tagihan, tempat_tinggal, transportasi |
| `payment_mode_*` | int | One-hot: ewallet, kartu, qris, tunai |
| `location_*` | int | One-hot: bandung, denpasar, jakarta, makassar, medan, palembang, semarang, surabaya, unknown, yogyakarta |
| `day_of_week_*` | int | One-hot: Monday, Tuesday, Wednesday, Thursday, Saturday, Sunday |

---

## Environment Variables

| Variable | Keterangan | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (wajib untuk genai_api) | - |
| `EXPENSE_API_URL` | URL expense prediction service | `http://localhost:8000` |
| `MODEL_DIR` | Folder model artifacts | `model_exports` |
| `LOGS_DIR` | Folder TensorBoard training logs | `logs` |

---

## Catatan

- Semua nilai fitur harus sudah ternormalisasi sesuai scaler dari pipeline Data Science
- Minimal baris input = `window_size` (lihat `model_config.json`)
- Field yang tidak disebutkan dalam request akan default ke 0
- `api_key` di body request bersifat opsional jika `GEMINI_API_KEY` sudah diset sebagai environment variable
- TensorBoard logs tersimpan di folder `logs/` — bisa dibuka visual via `tensorboard --logdir logs/` atau via API endpoint `/training-metrics`