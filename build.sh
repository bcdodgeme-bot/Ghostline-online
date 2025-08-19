#!/usr/bin/env bash
set -e  # Exit on any error

echo "🔧 Starting build process..."

# Update package lists
echo "📦 Updating package lists..."
apt-get update

# Install Tesseract OCR
echo "🔍 Installing Tesseract OCR..."
apt-get install -y tesseract-ocr tesseract-ocr-eng

# Verify Tesseract installation
echo "✅ Verifying Tesseract installation..."
tesseract --version

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

echo "🎉 Build process complete!"