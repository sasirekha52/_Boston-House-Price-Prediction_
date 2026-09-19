@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

if not exist ".venv\.requirements-installed" (
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed.
        exit /b 1
    )
    type nul > ".venv\.requirements-installed"
)

if not exist "models\house_price_model.pkl" (
    echo Training model because no trained model was found...
    python -m src.train
    if errorlevel 1 (
        echo Training failed.
        exit /b 1
    )
)

python -m streamlit run app.py
