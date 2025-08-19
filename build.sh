#!/usr/bin/env bash
set -e  # Exit on any error

echo "🔧 Starting build process..."

# Update package lists
echo "📦 Updating package lists..."
apt-get update

# Install Tesseract OCR
echo "🔍 Installing Tesseract OCR..."
apt-get install -y tesseract-ocr tesseract-ocr-eng

# Find where Tesseract was installed and add to PATH
echo "🔍 Finding Tesseract installation path..."
which tesseract || echo "Tesseract not found in current PATH"
find /usr -name "tesseract" -type f 2>/dev/null || echo "Tesseract binary not found"

# Try common installation paths
export PATH="/usr/bin:/usr/local/bin:$PATH"

# Verify Tesseract installation
echo "✅ Verifying Tesseract installation..."
tesseract --version

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

echo "🎉 Build process complete!"