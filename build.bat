@echo off
REM Local build script for testing (Windows)
REM Usage: build.bat [python-version]

setlocal enabledelayedexpansion
set PYTHON_VERSION=%1
if "!PYTHON_VERSION!"=="" set PYTHON_VERSION=3.11

echo 🔨 Building Cell Infiltrations with Python !PYTHON_VERSION!...

REM Check if Python is available
python !PYTHON_VERSION! --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Python !PYTHON_VERSION! not found
    exit /b 1
)

REM Create virtual environment if not exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python !PYTHON_VERSION! -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo 📚 Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

REM Build
echo 🏗️  Building executable...
pyinstaller build.spec

REM Verify build
if exist "dist\Cell Infiltrations.exe" (
    echo ✅ Build successful!
    echo.
    echo 📂 Executable location:
    dir "dist\Cell Infiltrations.exe"
    echo.
    echo 🚀 To run the executable:
    echo    "dist\Cell Infiltrations.exe"
) else (
    echo ❌ Build failed
    exit /b 1
)
