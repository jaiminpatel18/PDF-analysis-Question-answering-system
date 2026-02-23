Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " PDF Analysis Question Answering System" -ForegroundColor Cyan
Write-Host " Windows Setup Script (PowerShell)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r req.txt

# Install faiss-cpu separately
pip install faiss-cpu

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host " Setup complete! Starting the application..." -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "NOTE: If you need OCR support, install Tesseract OCR for Windows:" -ForegroundColor Yellow
Write-Host "  https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
Write-Host ""

# Run the Streamlit app
streamlit run app.py
