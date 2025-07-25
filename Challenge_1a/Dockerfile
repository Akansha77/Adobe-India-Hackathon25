FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies (minimal, only if needed)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies (PyMuPDF for PDF processing)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir PyMuPDF==1.23.24

# Copy the final submission processor
COPY final_precision_processor.py .

# Create required directories
RUN mkdir -p /app/input /app/output

# Set default command with proper input/output paths
CMD ["python", "final_precision_processor.py", "--input", "/app/input", "--output", "/app/output"] 