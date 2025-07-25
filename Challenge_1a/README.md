# Adobe India Hackathon 2025 - Round 1A
## PDF Document Intelligence System

### 🎯 Challenge Description
Extract titles and hierarchical headings from PDF documents with high precision and recall.

### 📋 System Requirements & Compliance
✅ **CPU-only processing** (no GPU dependencies)  
✅ **Processing time** <10 seconds per document  
✅ **Model size** <200MB (PyMuPDF only)  
✅ **Universal processing** (no hardcoded document-specific logic)  
✅ **Docker containerization** with proper I/O handling  
✅ **Offline operation** (no internet/API calls)  

### 🚀 Current Performance Metrics
- **Title Accuracy:** 60.0% (3/5 files)
- **Heading Precision:** 40.0%
- **Heading Recall:** 48.3%
- **F1-Score:** 43.8%
- **Processing Speed:** 0.1s average per file

### 🏗️ Architecture
- **Dynamic Font Clustering:** Automatic heading font detection
- **Semantic Title Scoring:** Content-aware title extraction
- **Layout Analysis:** Whitespace and position-based filtering
- **Header/Footer Detection:** Automatic removal of repeated elements
- **Multiline Title Combination:** Intelligent text grouping

### 📁 Project Structure
```
Challenge_1a/
├── final_precision_processor.py    # Main PDF processor
├── evaluate_metrics.py             # Performance evaluation
├── Dockerfile                      # Container configuration
├── README.md                       # This file
├── sample_dataset/                 # Test data
│   ├── pdfs/                      # Input PDF files
│   ├── outputs/                   # Ground truth JSON
│   └── schema/                    # Output schema
└── model_outputs/                 # Generated results
```

### 🛠️ Usage

#### Local Execution
```bash
python final_precision_processor.py --input sample_dataset/pdfs --output model_outputs
```

#### With Debug Output
```bash
python final_precision_processor.py --input sample_dataset/pdfs --output model_outputs --debug
```

#### Docker Execution
```bash
# Build container
docker build -t pdf-processor .

# Run processing
docker run -v $(pwd)/sample_dataset:/input -v $(pwd)/model_outputs:/output pdf-processor
```

#### Performance Evaluation
```bash
python evaluate_metrics.py
```

### 📊 Output Format
Each PDF generates a JSON file with:
```json
{
    "title": "Document Title ",
    "outline": [
        {
            "level": "H1",
            "text": "Heading Text ",
            "page": 1
        }
    ]
}
```

### 🔧 Algorithm Features

#### Title Extraction
- **Multiline Combination:** Intelligently combines title segments
- **Semantic Scoring:** Content-aware confidence calculation
- **Position Analysis:** Layout-based filtering
- **Format Recognition:** Generic pattern detection

#### Heading Detection
- **Font Clustering:** Dynamic analysis of document typography
- **Confidence Scoring:** Multi-factor heading validation
- **Hierarchy Mapping:** Automatic level assignment (H1-H4)
- **Noise Filtering:** Header/footer and metadata removal

### 🎪 Technical Highlights
- **Zero Hardcoding:** Fully generic processing pipeline
- **Fast Processing:** Sub-second per document
- **Memory Efficient:** Minimal resource usage
- **Error Resilient:** Graceful handling of malformed PDFs
- **Scalable Design:** Processes any PDF format/size

### 📈 Performance Optimization
- Dynamic confidence thresholding
- Layout-aware whitespace analysis
- Font frequency-based filtering
- Semantic content validation
- Multi-factor scoring algorithms

### 🔍 Testing
The system is evaluated against a diverse dataset including:
- Government forms
- Technical documentation
- Proposal documents
- Educational materials
- Presentation slides

### 📝 Dependencies
- **PyMuPDF (fitz):** PDF processing library
- **Python 3.8+:** Runtime environment
- **Standard libraries:** json, re, pathlib, collections, argparse

### 🏆 Adobe Hackathon Compliance Statement
This system strictly adheres to all Adobe Hackathon guidelines:
- No document-specific hardcoded logic
- Generic pattern-based processing only
- Universal applicability across document types
- CPU-only operation with fast processing
- Containerized deployment ready