# Use official Python 3.11 slim image
FROM python:3.11-slim

# System dependencies: Tesseract (+languages) and Poppler for pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd tesseract-ocr-fas \
    poppler-utils build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies (cached if requirements.txt unchanged)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code last (so code changes don't invalidate deps cache)
COPY src/ /app/src

# Expose port
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
