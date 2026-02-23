@echo off
echo =============================================
echo  PDF Analysis Question Answering System
echo  Windows Setup Script
echo =============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r req.txt

REM Install faiss-cpu separately (just in case)
pip install faiss-cpu

echo.
echo =============================================
echo  Setup complete! Starting the application...
echo =============================================
echo.
echo NOTE: If you need OCR support, install Tesseract OCR for Windows:
echo   https://github.com/UB-Mannheim/tesseract/wiki
echo   After installing, add it to your PATH or set TESSERACT_CMD in .env
echo.

REM Run the Streamlit app
streamlit run app.py

pause
