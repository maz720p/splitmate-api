@echo off
echo SplitMate API - Setup dan Jalankan
echo ----------------------------------

:: Cek Python tersedia
python --version >nul 2>&1
if errorlevel 1 (
    echo Python tidak ditemukan. Install Python dulu dari https://python.org
    pause
    exit /b 1
)

:: Install dependencies
echo [1/3] Install dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Gagal install dependencies. Coba jalankan CMD sebagai Administrator.
    pause
    exit /b 1
)

:: Set Gemini API key jika belum ada
if "%GEMINI_API_KEY%"=="" (
    echo.
    echo GEMINI_API_KEY belum diset.
    set /p GEMINI_API_KEY="Masukkan Gemini API Key kamu: "
)

:: Jalankan Expense Prediction API
echo.
echo [2/3] Menjalankan Expense Prediction API di port 8000...
start "Expense API" cmd /k "uvicorn expense_api:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak >nul

:: Jalankan GenAI Insight API
echo [3/3] Menjalankan GenAI Insight API di port 8001...
start "GenAI API" cmd /k "set GEMINI_API_KEY=%GEMINI_API_KEY% && uvicorn genai_api:app --host 0.0.0.0 --port 8001 --reload"

echo.
echo Kedua API sudah berjalan!
echo Expense API : http://localhost:8000/docs
echo GenAI API   : http://localhost:8001/docs
echo.
echo TensorBoard : tensorboard --logdir logs/
echo ----------------------------------
pause
