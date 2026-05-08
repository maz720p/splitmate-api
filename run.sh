#!/bin/bash
echo "========================================"
echo " SplitMate API - Setup dan Jalankan"
echo "========================================"

# Install dependencies
echo "[1/3] Install dependencies..."
pip install -r requirements.txt

# Set API key jika belum ada
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "[!] ANTHROPIC_API_KEY belum diset."
    read -p "Masukkan Anthropic API Key kamu: " ANTHROPIC_API_KEY
    export ANTHROPIC_API_KEY
fi

# Jalankan kedua service
echo ""
echo "[2/3] Menjalankan Expense Prediction API di port 8000..."
uvicorn expense_api:app --host 0.0.0.0 --port 8000 --reload &
PID1=$!

sleep 2

echo "[3/3] Menjalankan GenAI Insight API di port 8001..."
uvicorn genai_api:app --host 0.0.0.0 --port 8001 --reload &
PID2=$!

echo ""
echo "========================================"
echo " Kedua API sudah berjalan!"
echo " Expense API : http://localhost:8000/docs"
echo " GenAI API   : http://localhost:8001/docs"
echo "========================================"
echo " Tekan Ctrl+C untuk stop semua."
echo "========================================"

# Tunggu sampai di-stop
wait $PID1 $PID2
