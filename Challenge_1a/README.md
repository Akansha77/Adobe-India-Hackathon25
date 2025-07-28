# Challenge 1a: PDF Document Intelligence

## 🎯 Objective
Extract titles and hierarchical headings from PDF documents with high precision and recall.

## � Performance Results
- **Title Accuracy:** 60.0%
- **Heading Precision:** 31.9%
- **Heading Recall:** 91.4%
- **F1-Score:** 47.3%

## 🛠️ Usage
```bash
# Process PDFs
python main.py sample_dataset/pdfs -o model_outputs

# Evaluate results
python evaluate.py model_outputs sample_dataset/outputs
```

## 📁 Structure
```
Challenge_1a/
├── main.py              # PDF processor
├── evaluate.py          # Evaluation script
├── sample_dataset/      # Test data & ground truth
└── model_outputs/       # Generated results
```
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