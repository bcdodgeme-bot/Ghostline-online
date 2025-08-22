FROM python:3.11-slim

# Install system dependencies including Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .


# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "app:app", "-w", "2", "-k", "gthread", "--threads", "8", "--timeout", "120", "--bind", "0.0.0.0:5000"]

