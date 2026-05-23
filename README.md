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
└── logs/                         <-- TensorBoard training logs (rename dari tb_logs/ setelah export)
    └── 20260520-221031/
        ├── train/
        ├── validation/
        ├── gt_train/             <-- dari GradientTape loop
        └── gt_val/               <-- dari GradientTape loop
```

> **Catatan:** Notebook menyimpan log ke `tb_logs/` secara default. Setelah download
> `tensorboard_logs.zip`, extract dan rename folder `tb_logs/` menjadi `logs/`
> (sesuai default `LOGS_DIR` di `expense_api.py` dan `tensorboard_utils.py`),
> atau set env var `LOGS_DIR=tb_logs` saat menjalankan service.

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

Folder `logs/` berisi training logs dari sesi training model BiLSTM + Attention v4:

| Run | Keterangan |
|---|---|
| `20260520-221031` | Sesi training utama (Keras `model.fit` callback) |
| `gradient_tape_20260511-075701` | Sesi training dengan custom `GradientTape` loop |

Run `20260520-221031` memiliki 4 split: `train`, `validation`, `gt_train`, `gt_val`.

### Membuka TensorBoard (visual)

```bash
tensorboard --logdir logs/
```

Lalu buka http://localhost:6006

### Membaca metrik via API

`tensorboard_utils.py` membaca isi event files secara programatik dan dipakai oleh endpoint berikut:

| Endpoint | Keterangan |
|---|---|
| `GET /training-logs` | Daftar runs + path event files |
| `GET /training-metrics` | Ringkasan metrik semua run (tanpa series) |
| `GET /training-metrics/{run_name}` | Metrik lengkap + series per step untuk satu run |

Contoh response `GET /training-metrics/20260520-221031`:

```json
{
  "status": "ok",
  "run_name": "20260520-221031",
  "splits": ["train", "validation", "gt_train", "gt_val"],
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
metrics = tensorboard_utils.get_run_metrics("20260520-221031")
```

### Mengganti lokasi folder logs

Set environment variable `LOGS_DIR` (berlaku untuk `expense_api.py` dan `tensorboard_utils.py`):

```bash
LOGS_DIR=/path/lain uvicorn expense_api:app --port 8000 --reload

# Contoh jika logs masih di tb_logs/ (belum di-rename):
LOGS_DIR=tb_logs uvicorn expense_api:app --port 8000 --reload
```

---

## 1. expense_api.py

FastAPI untuk prediksi pengeluaran menggunakan model BiLSTM + Attention v4. Berjalan di port 8000.

Endpoints:
- `GET /health` - status model dan ketersediaan logs
- `GET /model-info` - metrik model (MAE, akurasi, window size, feature list)
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
      "amount_idr": 250000,
      "month": 5,
      "is_weekend": 0,
      "year": 2026,
      "category_makanan": 1,
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
  "n_features": 21,
  "predictions": [
    { "step": 1, "prediksi_norm": 0.032, "prediksi_idr": 850000.0, "prediksi_fmt": "Rp 850,000" }
  ],
  "mae_normalized": 0.0158,
  "accuracy_pct": 97.9,
  "inference_note": "Input: X_raw (IDR mentah) -> scaler_X.transform(X_raw). Output: scaler_y.inverse_transform(y_pred) -> np.expm1 -> IDR"
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
  "rows": [
    {
      "amount_idr": 250000,
      "month": 5,
      "is_weekend": 0,
      "year": 2026,
      "category_makanan": 1,
      "day_of_week_Monday": 1
    }
  ],
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

## Daftar Field FeatureRow (Model v4 — 21 Fitur)

> **Penting:** Model v4 **tidak** menggunakan `payment_mode`, `location`,
> `transaction_type_income`, `category_freelance`, atau `category_gaji`.
> `amount_idr` dikirim dalam **IDR mentah** (bukan dibagi juta), normalisasi
> dilakukan otomatis oleh `scaler_X` di backend.

| Field | Tipe | Keterangan |
|---|---|---|
| `amount_idr` | float | Jumlah transaksi dalam IDR mentah (misal: `250000`) |
| `month` | float | Bulan 1–12 |
| `is_weekend` | int | 1 jika weekend, 0 jika tidak |
| `year` | float | Tahun (2022–2026) |
| `category_makanan` | int | One-hot kategori |
| `category_tempat_tinggal` | int | One-hot kategori |
| `category_transportasi` | int | One-hot kategori |
| `category_tagihan` | int | One-hot kategori |
| `category_hiburan` | int | One-hot kategori |
| `category_kesehatan` | int | One-hot kategori |
| `category_pendidikan` | int | One-hot kategori |
| `category_tabungan` | int | One-hot kategori |
| `category_lainnya` | int | One-hot kategori |
| `category_investasi` | int | One-hot kategori |
| `day_of_week_Friday` | int | One-hot hari (semua 7 hari eksplisit, tidak ada reference category) |
| `day_of_week_Monday` | int | One-hot hari |
| `day_of_week_Saturday` | int | One-hot hari |
| `day_of_week_Sunday` | int | One-hot hari |
| `day_of_week_Thursday` | int | One-hot hari |
| `day_of_week_Tuesday` | int | One-hot hari |
| `day_of_week_Wednesday` | int | One-hot hari |

Field yang tidak disebutkan dalam request akan default ke 0.

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

- `amount_idr` dikirim dalam IDR mentah — normalisasi dilakukan otomatis oleh backend via `scaler_X`
- Minimal baris input = `window_size` (10, lihat `model_config.json`) — jika kurang, zero-padding otomatis ditambahkan di depan
- `api_key` di body request `/insight` bersifat opsional jika `GEMINI_API_KEY` sudah diset sebagai environment variable
- TensorBoard logs tersimpan di folder `logs/` — bisa dibuka visual via `tensorboard --logdir logs/` atau via API endpoint `/training-metrics`
- Model v4 MAE normalized: **0.0158**, Accuracy (NAE ≤ 0.05): **97.9%**
