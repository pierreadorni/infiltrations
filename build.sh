#!/bin/bash
# Local build script for testing (macOS/Linux)
# Usage: ./build.sh [python-version]

set -e  # Exit on error

PYTHON_VERSION=${1:-"3.11"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔨 Building Cell Infiltrations with Python $PYTHON_VERSION..."

# Check if Python is available
if ! command -v python$PYTHON_VERSION &> /dev/null; then
    echo "❌ Python $PYTHON_VERSION not found"
    exit 1
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python$PYTHON_VERSION -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# Build
echo "🏗️  Building executable..."
pyinstaller --onefile --noconsole build.spec

# Verify build
if [ -f "dist/Cell Infiltrations" ]; then
    echo "✅ Build successful!"
    echo ""
    echo "📂 Executable location:"
    ls -lh "dist/Cell Infiltrations"
    echo ""
    echo "🚀 To run the executable:"
    echo "   ./dist/Cell\\ Infiltrations"
else
    echo "❌ Build failed"
    exit 1
fi
