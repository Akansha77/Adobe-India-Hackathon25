FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir PyPDF2==3.0.1 pdfplumber==0.10.3

# Copy the processing script
COPY process_pdfs.py .

# Run the script
CMD ["python", "process_pdfs.py"] 