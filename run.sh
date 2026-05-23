#!/bin/bash
echo "SplitMate API - Setup dan Jalankan"
echo "----------------------------------"

# Cek Python tersedia
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Python tidak ditemukan. Install Python dulu dari https://python.org"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)

# Install dependencies
echo "[1/3] Install dependencies..."
$PYTHON -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Gagal install dependencies."
    exit 1
fi

# Set Gemini API key jika belum ada
if [ -z "$GEMINI_API_KEY" ]; then
    echo ""
    echo "GEMINI_API_KEY belum diset."
    read -p "Masukkan Gemini API Key kamu: " GEMINI_API_KEY
    export GEMINI_API_KEY
fi

# Jalankan Expense Prediction API
echo ""
echo "[2/3] Menjalankan Expense Prediction API di port 8000..."
uvicorn expense_api:app --host 0.0.0.0 --port 8000 --reload &
PID1=$!

sleep 2

# Jalankan GenAI Insight API
echo "[3/3] Menjalankan GenAI Insight API di port 8001..."
GEMINI_API_KEY=$GEMINI_API_KEY uvicorn genai_api:app --host 0.0.0.0 --port 8001 --reload &
PID2=$!

echo ""
echo "Kedua API sudah berjalan!"
echo "Expense API : http://localhost:8000/docs"
echo "GenAI API   : http://localhost:8001/docs"
echo ""
echo "TensorBoard : tensorboard --logdir logs/"
echo "----------------------------------"
echo "Tekan Ctrl+C untuk stop semua."

trap "kill $PID1 $PID2 2>/dev/null; exit 0" INT TERM
wait $PID1 $PID2
