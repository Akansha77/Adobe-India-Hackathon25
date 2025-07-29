# Adobe India Hackathon 2025

## 🎯 Overview
PDF intelligence solutions for document processing and content extraction challenges. This project implements advanced PDF analysis capabilities using PyMuPDF for extracting structured information from documents.

## � Key Features
- **CPU-only processing** with optimized performance
- **Universal PDF processing** (no hardcoded logic)
- **Persona-based content extraction** for different use cases
- **Hierarchical heading detection** with font analysis
- **Multi-collection processing** across domains

## �📊 Challenge Results

### Challenge 1a: PDF Document Intelligence
Extract titles and hierarchical headings from PDFs
- **Title Accuracy:** 60.0%
- **Heading Precision:** 31.9%
- **Heading Recall:** 91.4%
- **F1-Score:** 47.3%
- **Processing Speed:** ~0.1s per file

### Challenge 1b: Multi-Collection PDF Intelligence  
Extract persona-based content from PDF collections
- **Average Precision:** 50.0%
- **Average Recall:** 40.0%
- **Average F1-Score:** 44.4%
- **Best Collection:** Travel Planner (88.9%)

## �️ Technologies Used
- **Python 3.x** - Core programming language
- **PyMuPDF (fitz)** - PDF processing and text extraction
- **JSON** - Data serialization and output format
- **Docker** - Containerization support

## �🚀 Quick Start
```bash
# Run Challenge 1a
cd Challenge_1a
python main.py sample_dataset/pdfs -o model_outputs
python evaluate.py model_outputs sample_dataset/outputs

# Run Challenge 1b
cd Challenge_1b
python run_collections.py
python evaluate_challenge1b.py
```

## 📁 Project Structure
```
Adobe-India-Hackathon25/
├── Challenge_1a/           # PDF title & heading extraction
│   ├── main.py            # Core PDF processor
│   ├── evaluate.py        # Performance evaluation
│   ├── sample_dataset/    # Test data & ground truth
│   └── Final_outputs/     # Generated results
└── Challenge_1b/          # Multi-collection PDF intelligence
    ├── run_collections.py # Main collection runner
    ├── evaluate_challenge1b.py # Evaluation script
    ├── Collection 1/      # Travel planning domain
    ├── Collection 2/      # HR training domain
    └── Collection 3/      # Food service domain
```

## 📋 Requirements
- Python 3.7+
- PyMuPDF (`pip install PyMuPDF`)
- Standard Python libraries (json, argparse, pathlib, etc.)

## 🎯 Challenge Objectives
1. **Challenge 1a**: Build a robust PDF title and heading extraction system with high precision and recall
2. **Challenge 1b**: Develop persona-aware content extraction that adapts to different user roles and tasks

## 📈 Performance Insights
- Challenge 1a excels in recall (91.4%) but needs precision improvement
- Challenge 1b shows strong performance for travel planning scenarios
- Both challenges demonstrate CPU-efficient processing capabilities
