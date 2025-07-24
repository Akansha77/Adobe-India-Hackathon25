import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import PyPDF2
import pdfplumber
import unicodedata

class PDFProcessor:
    def __init__(self):
        # Common heading patterns (will be refined based on content analysis)
        self.heading_patterns = [
            # Numbered headings
            r'^(\d+\.?\s+.*)$',  # "1. Introduction", "1 Introduction"
            r'^(\d+\.\d+\.?\s+.*)$',  # "1.1 Overview", "1.1. Overview"
            r'^(\d+\.\d+\.\d+\.?\s+.*)$',  # "1.1.1 Details"
            
            # Lettered headings
            r'^([A-Z]\.?\s+.*)$',  # "A. Section"
            r'^([a-z]\.?\s+.*)$',  # "a. subsection"
            
            # Roman numerals
            r'^([IVX]+\.?\s+.*)$',  # "I. Introduction"
            r'^([ivx]+\.?\s+.*)$',  # "i. introduction"
            
            # Common section headers
            r'^(TABLE OF CONTENTS?)$',
            r'^(CONTENTS?)$',
            r'^(ACKNOWLEDGEMENTS?)$',
            r'^(ABSTRACT)$',
            r'^(INTRODUCTION)$',
            r'^(CONCLUSION)$',
            r'^(REFERENCES?)$',
            r'^(BIBLIOGRAPHY)$',
            r'^(APPENDIX.*)$',
            r'^(SUMMARY)$',
            r'^(BACKGROUND)$',
            r'^(OVERVIEW)$',
            r'^(REVISION HISTORY)$',
        ]
        
        # Title extraction patterns
        self.title_stop_words = {
            'table', 'contents', 'page', 'chapter', 'section', 'appendix',
            'figure', 'revision', 'history', 'acknowledgements', 'abstract'
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\-\(\)\[\],:;!?\'\"&]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text

    def extract_title_from_metadata(self, pdf_path: Path) -> str:
        """Extract title from PDF metadata"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                
                if metadata and metadata.title:
                    title = self.clean_text(str(metadata.title))
                    if len(title) > 3 and len(title) < 200:
                        return title
        except Exception:
            pass
        return None

    def extract_title_from_content(self, pdf_path: Path) -> str:
        """Extract title from PDF content using pdfplumber"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                title_candidates = []
                
                # Check first 3 pages for title
                for page_num in range(min(3, len(pdf.pages))):
                    page = pdf.pages[page_num]
                    
                    # Extract text with font information
                    chars = page.chars
                    if not chars:
                        continue
                    
                    # Group characters by line based on y-coordinate
                    lines = {}
                    for char in chars:
                        y = round(char['y0'], 1)
                        if y not in lines:
                            lines[y] = []
                        lines[y].append(char)
                    
                    # Sort lines by y-coordinate (top to bottom)
                    sorted_lines = sorted(lines.items(), key=lambda x: -x[0])
                    
                    for y, line_chars in sorted_lines[:10]:  # Check top 10 lines per page
                        # Sort characters by x-coordinate
                        line_chars.sort(key=lambda x: x['x0'])
                        
                        # Extract text and font size
                        line_text = ''.join(char['text'] for char in line_chars)
                        line_text = self.clean_text(line_text)
                        
                        if not line_text or len(line_text) < 5 or len(line_text) > 200:
                            continue
                        
                        # Skip lines that look like headers/footers
                        line_lower = line_text.lower()
                        if any(word in line_lower for word in self.title_stop_words):
                            continue
                        
                        # Skip lines with page numbers
                        if re.match(r'^\d+$', line_text.strip()):
                            continue
                        
                        # Get average font size for this line
                        font_sizes = [char.get('size', 12) for char in line_chars if char.get('size')]
                        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                        
                        title_candidates.append((line_text, avg_font_size, page_num))
                
                if title_candidates:
                    # Sort by font size (larger fonts likely to be title) and page (earlier pages preferred)
                    title_candidates.sort(key=lambda x: (-x[1], x[2]))
                    return title_candidates[0][0]
                    
        except Exception as e:
            print(f"Error extracting title from content: {e}")
            
        return None

    def extract_title(self, pdf_path: Path) -> str:
        """Extract document title"""
        # Try metadata first
        title = self.extract_title_from_metadata(pdf_path)
        if title:
            return title
        
        # Try content extraction
        title = self.extract_title_from_content(pdf_path)
        if title:
            return title
        
        # Fallback
        return f"Document: {pdf_path.stem}"

    def extract_headings_with_pdfplumber(self, pdf_path: Path) -> List[Tuple[str, int, str]]:
        """Extract headings using pdfplumber"""
        headings = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    chars = page.chars
                    if not chars:
                        continue
                    
                    # Group characters by line
                    lines = {}
                    for char in chars:
                        y = round(char['y0'], 1)
                        if y not in lines:
                            lines[y] = []
                        lines[y].append(char)
                    
                    # Process each line
                    for y, line_chars in lines.items():
                        line_chars.sort(key=lambda x: x['x0'])
                        
                        # Extract text and font information
                        line_text = ''.join(char['text'] for char in line_chars)
                        line_text = self.clean_text(line_text)
                        
                        if not line_text or len(line_text) < 2:
                            continue
                        
                        # Get font information
                        font_sizes = [char.get('size', 12) for char in line_chars if char.get('size')]
                        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                        
                        # Check if this line is a heading
                        heading_level = self.determine_heading_level(line_text, avg_font_size, False)
                        if heading_level:
                            headings.append((line_text, page_num + 1, heading_level))
                            
        except Exception as e:
            print(f"Error extracting headings with pdfplumber: {e}")
        
        return headings

    def extract_headings_with_pypdf2(self, pdf_path: Path) -> List[Tuple[str, int, str]]:
        """Extract headings using PyPDF2 as fallback"""
        headings = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    lines = text.split('\n')
                    for line in lines:
                        line = self.clean_text(line)
                        if not line or len(line) < 2:
                            continue
                        
                        # Simple heading detection based on patterns
                        heading_level = self.determine_heading_level(line, 12, False)
                        if heading_level:
                            headings.append((line, page_num + 1, heading_level))
                            
        except Exception as e:
            print(f"Error extracting headings with PyPDF2: {e}")
        
        return headings

    def determine_heading_level(self, text: str, font_size: float, is_bold: bool) -> Optional[str]:
        """Determine if text is a heading and its level"""
        text_upper = text.upper()
        text_clean = text.strip()
        
        # Skip very long lines (unlikely to be headings)
        if len(text_clean) > 150:
            return None
        
        # Check for explicit patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return self.classify_heading_level(text_clean)
        
        # Font-based detection for larger text
        if font_size > 13:
            # Additional checks for common heading characteristics
            if (len(text_clean) < 100 and  # Not too long
                not text_clean.endswith('.') and  # Doesn't end with period
                not re.match(r'.*\d{4}.*', text_clean) and  # No years
                not re.match(r'^page \d+', text_clean.lower())):  # Not page numbers
                return self.classify_heading_level(text_clean)
        
        return None

    def classify_heading_level(self, text: str) -> str:
        """Classify heading into H1, H2, H3 based on content patterns"""
        text_clean = text.strip()
        
        # H1 patterns (major sections)
        h1_patterns = [
            r'^\d+\.?\s+',  # "1. Introduction"
            r'^[A-Z]+\.?\s+',  # "INTRODUCTION"
            r'^(TABLE OF CONTENTS?|CONTENTS?)$',
            r'^(ACKNOWLEDGEMENTS?|ABSTRACT|INTRODUCTION|CONCLUSION|REFERENCES?|BIBLIOGRAPHY|SUMMARY|BACKGROUND|OVERVIEW|REVISION HISTORY)$'
        ]
        
        # H2 patterns (subsections)
        h2_patterns = [
            r'^\d+\.\d+\.?\s+',  # "1.1 Overview"
            r'^[A-Z]\.\d+\.?\s+',  # "A.1 Section"
        ]
        
        # H3 patterns (sub-subsections)
        h3_patterns = [
            r'^\d+\.\d+\.\d+\.?\s+',  # "1.1.1 Details"
            r'^[a-z]\)?\s+',  # "a) subsection"
        ]
        
        for pattern in h3_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return "H3"
        
        for pattern in h2_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return "H2"
        
        for pattern in h1_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return "H1"
        
        # Default classification
        if len(text_clean) < 50:  # Short text, likely heading
            if any(word in text_clean.upper() for word in ['CHAPTER', 'SECTION', 'PART']):
                return "H1"
            elif '.' in text_clean and len(text_clean.split('.')) <= 3:
                return "H2"
            else:
                return "H1"  # Default to H1 for unclassified headings
        
        return "H1"

    def process_pdf(self, pdf_path: Path) -> Dict:
        """Process a single PDF and extract structured data"""
        try:
            print(f"  Extracting title...")
            title = self.extract_title(pdf_path)
            
            print(f"  Extracting headings...")
            # Try pdfplumber first, fallback to PyPDF2
            headings = self.extract_headings_with_pdfplumber(pdf_path)
            if not headings:
                headings = self.extract_headings_with_pypdf2(pdf_path)
            
            # Format outline
            outline = []
            for text, page, level in headings:
                outline.append({
                    "level": level,
                    "text": text,
                    "page": page
                })
            
            return {
                "title": title,
                "outline": outline
            }
            
        except Exception as e:
            print(f"  Error processing {pdf_path}: {str(e)}")
            return {
                "title": f"Error processing {pdf_path.name}",
                "outline": []
            }

def process_pdfs():
    """Main processing function"""
    # For local testing, use sample data
    if os.path.exists("sample_dataset"):
        input_dir = Path("sample_dataset/pdfs")
        output_dir = Path("test_output")
    else:
        # Docker environment paths
        input_dir = Path("/app/input")
        output_dir = Path("/app/output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        
        # Process the PDF
        result = processor.process_pdf(pdf_file)
        
        # Create output JSON file
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"Completed {pdf_file.name} -> {output_file.name} (Found {len(result['outline'])} headings)")

if __name__ == "__main__":
    print("Starting PDF processing...")
    process_pdfs() 
    print("Completed PDF processing!")
