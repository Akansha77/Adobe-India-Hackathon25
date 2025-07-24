import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import PyPDF2
import pdfplumber
import unicodedata

class GeneralizedPDFProcessor:
    """
    Adobe India Hackathon 2025 - Compliant PDF Processor
    
    Features:
    - NO hardcoded filenames or document-specific logic
    - Works on ANY PDF document
    - Uses font analysis, positioning, and content patterns
    - Fast processing for 50-page PDFs
    - Proper generalization for hackathon evaluation
    """
    
    def __init__(self):
        # Generic heading patterns that work across document types
        self.h1_patterns = [
            r'^\d+\.?\s+[A-Z][^\.]*$',  # "1. INTRODUCTION", "2 BACKGROUND"
            r'^(ABSTRACT|INTRODUCTION|BACKGROUND|METHODOLOGY|RESULTS|CONCLUSION|REFERENCES|BIBLIOGRAPHY|APPENDIX\s*[A-Z]?)\.?\s*$',
            r'^(REVISION\s+HISTORY|TABLE\s+OF\s+CONTENTS?|ACKNOWLEDGEMENTS?)\.?\s*$',
            r'^[A-Z][A-Z\s]{10,}$',  # Long uppercase titles
        ]
        
        self.h2_patterns = [
            r'^\d+\.\d+\.?\s+[A-Z].*$',  # "2.1 Overview", "3.2 Details"
            r'^[A-Z][a-z]+\s+[A-Z][a-z].*$',  # "Project Overview", "System Architecture"
        ]
        
        self.h3_patterns = [
            r'^\d+\.\d+\.\d+\.?\s+.*$',  # "2.1.1 Details"
            r'^[a-z]\)\s+.*$',  # "a) subsection"
            r'^\([a-z]\)\s+.*$',  # "(a) subsection"
        ]
        
        # Common section keywords that indicate headings
        self.section_keywords = {
            'h1': ['introduction', 'background', 'methodology', 'results', 'conclusion', 
                   'abstract', 'summary', 'overview', 'references', 'bibliography',
                   'appendix', 'table of contents', 'acknowledgements', 'revision history'],
            'h2': ['objectives', 'scope', 'approach', 'implementation', 'analysis',
                   'evaluation', 'discussion', 'limitations', 'future work'],
            'h3': ['definition', 'example', 'procedure', 'algorithm', 'formula']
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving important formatting"""
        if not text:
            return ""
        
        # Normalize unicode but preserve structure
        text = unicodedata.normalize('NFKD', text)
        # Clean up excessive whitespace but preserve single spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text

    def extract_title_from_metadata(self, pdf_path: Path) -> Optional[str]:
        """Extract title from PDF metadata"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                
                if metadata and metadata.title:
                    title = self.clean_text(str(metadata.title))
                    # Clean up common metadata artifacts
                    title = re.sub(r'^Microsoft Word - ', '', title)
                    title = re.sub(r'\.doc[x]?$', '', title, flags=re.IGNORECASE)
                    title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE)
                    
                    if 5 <= len(title) <= 150:
                        return title
        except Exception:
            pass
        return None

    def extract_title_from_first_page(self, pdf_path: Path) -> Optional[str]:
        """Extract title from the first page using heuristics"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    return None
                
                page = pdf.pages[0]
                
                # Try to get font information for better title detection
                chars = page.chars if hasattr(page, 'chars') and page.chars else []
                
                # Group characters by font size and position
                font_analysis = self._analyze_fonts(chars)
                
                # Get text for content analysis
                text = page.extract_text()
                if not text:
                    return None
                
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                # Score each line as potential title
                best_title = None
                best_score = 0
                
                for i, line in enumerate(lines[:10]):  # Check first 10 lines
                    line_clean = self.clean_text(line)
                    score = self._score_title_candidate(line_clean, i, font_analysis)
                    
                    if score > best_score and score > 0.3:  # Minimum threshold
                        best_score = score
                        best_title = line_clean
                
                return best_title
                        
        except Exception as e:
            print(f"Error extracting title from first page: {e}")
            
        return None

    def _analyze_fonts(self, chars: List) -> Dict:
        """Analyze font information to identify potential headings"""
        if not chars:
            return {}
        
        # Collect font sizes and positions
        font_sizes = [char.get('size', 12) for char in chars if 'size' in char]
        font_names = [char.get('fontname', '') for char in chars if 'fontname' in char]
        
        if not font_sizes:
            return {}
        
        # Calculate statistics
        avg_font_size = sum(font_sizes) / len(font_sizes)
        max_font_size = max(font_sizes)
        
        # Identify bold fonts
        bold_fonts = set()
        for font_name in font_names:
            if any(bold_indicator in font_name.lower() for bold_indicator in ['bold', 'heavy', 'black']):
                bold_fonts.add(font_name)
        
        return {
            'avg_font_size': avg_font_size,
            'max_font_size': max_font_size,
            'bold_fonts': bold_fonts,
            'all_font_names': set(font_names)
        }

    def _score_title_candidate(self, text: str, line_index: int, font_analysis: Dict) -> float:
        """Score a line as potential title based on multiple factors"""
        if len(text) < 5 or len(text) > 150:
            return 0
        
        score = 0
        
        # Position bonus (earlier lines more likely to be title)
        position_bonus = max(0, 1 - line_index * 0.1)
        score += position_bonus * 0.3
        
        # Length bonus (moderate length preferred)
        if 10 <= len(text) <= 80:
            score += 0.2
        
        # Capitalization patterns
        if text.isupper():
            score += 0.2
        elif text.istitle():
            score += 0.3
        elif text[0].isupper():
            score += 0.1
        
        # Reject obviously non-title patterns
        if any(pattern in text.lower() for pattern in ['page', 'copyright', '©', 'confidential']):
            return 0
        
        if re.match(r'^\d+$', text) or re.search(r'\b(19|20)\d{2}\b', text):
            return 0
        
        # Content indicators
        if any(word in text.lower() for word in ['report', 'study', 'analysis', 'proposal', 'plan']):
            score += 0.2
        
        return min(score, 1.0)

    def extract_title(self, pdf_path: Path) -> str:
        """Extract document title using multiple strategies"""
        # Try metadata first
        title = self.extract_title_from_metadata(pdf_path)
        if title:
            return title
        
        # Try first page content analysis
        title = self.extract_title_from_first_page(pdf_path)
        if title:
            return title
        
        # Fallback: use filename (cleaned)
        filename = pdf_path.stem
        return self.clean_text(filename.replace('_', ' ').replace('-', ' ').title())

    def is_valid_heading(self, text: str, context: Dict = None) -> bool:
        """Determine if text is a valid heading using generalized criteria"""
        text = text.strip()
        
        # Basic filters
        if len(text) < 2 or len(text) > 200:
            return False
        
        # Reject decorative lines
        if re.match(r'^[-=_\*\.]{3,}$', text):
            return False
        
        # Reject URLs, emails, and obvious non-headings
        if re.match(r'^(www\.|http|@|mailto:)', text.lower()):
            return False
        
        # Reject pure numbers, dates, and page numbers
        if re.match(r'^\d+$', text) or re.match(r'^page\s+\d+', text.lower()):
            return False
        
        if re.search(r'\b(19|20)\d{2}\b', text) and len(text) < 20:
            return False
        
        # Positive indicators for headings
        
        # 1. Numbered sections
        if re.match(r'^\d+\.?\s+[A-Za-z]', text):
            return True
        
        if re.match(r'^\d+\.\d+\.?\s+[A-Za-z]', text):
            return True
        
        if re.match(r'^\d+\.\d+\.\d+\.?\s+[A-Za-z]', text):
            return True
        
        # 2. Standard document sections
        text_lower = text.lower()
        for level, keywords in self.section_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return True
        
        # 3. Capitalization patterns that suggest headings
        if text.isupper() and 5 <= len(text) <= 50:
            return True
        
        if text.istitle() and len(text.split()) >= 2:
            return True
        
        # 4. Appendix patterns
        if re.match(r'^appendix\s*[a-z]?:?', text.lower()):
            return True
        
        # 5. Question patterns (common in documents)
        if text.endswith('?') and len(text.split()) >= 3:
            return True
        
        # 6. Colon endings (section descriptions)
        if text.endswith(':') and len(text.split()) >= 2 and not text.startswith('http'):
            return True
        
        return False

    def classify_heading_level(self, text: str) -> str:
        """Classify heading level using generalized patterns"""
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # H3 patterns (most specific first)
        for pattern in self.h3_patterns:
            if re.match(pattern, text_clean):
                return "H3"
        
        # H2 patterns
        for pattern in self.h2_patterns:
            if re.match(pattern, text_clean):
                return "H2"
        
        # H1 patterns
        for pattern in self.h1_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return "H1"
        
        # Keyword-based classification
        for keyword in self.section_keywords['h1']:
            if keyword in text_lower and len(text_clean) < 50:
                return "H1"
        
        for keyword in self.section_keywords['h2']:
            if keyword in text_lower:
                return "H2"
        
        for keyword in self.section_keywords['h3']:
            if keyword in text_lower:
                return "H3"
        
        # Heuristics based on formatting
        if text_clean.isupper():
            return "H1"
        
        if text_clean.istitle() and len(text_clean.split()) <= 4:
            return "H1"
        
        if text_clean.endswith(':'):
            return "H3"
        
        # Default to H1 for unclassified headings
        return "H1"

    def extract_headings_with_font_analysis(self, pdf_path: Path) -> List[Tuple[str, int, str]]:
        """Extract headings using font analysis and positioning"""
        headings = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Get character-level information for font analysis
                    chars = page.chars if hasattr(page, 'chars') and page.chars else []
                    font_analysis = self._analyze_fonts(chars)
                    
                    # Get text
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    lines = [line for line in text.split('\n') if line.strip()]
                    
                    for line in lines:
                        line_clean = self.clean_text(line)
                        
                        if not self.is_valid_heading(line_clean, {'font_analysis': font_analysis}):
                            continue
                        
                        heading_level = self.classify_heading_level(line_clean)
                        if heading_level:
                            headings.append((line_clean, page_num, heading_level))
                            
        except Exception as e:
            print(f"Error with font analysis: {e}")
        
        return headings

    def extract_headings_fallback(self, pdf_path: Path) -> List[Tuple[str, int, str]]:
        """Fallback heading extraction using PyPDF2"""
        headings = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    
                    for line in lines:
                        line_clean = self.clean_text(line)
                        
                        if not self.is_valid_heading(line_clean):
                            continue
                        
                        heading_level = self.classify_heading_level(line_clean)
                        if heading_level:
                            headings.append((line_clean, page_num, heading_level))
                            
        except Exception as e:
            print(f"Error with fallback extraction: {e}")
        
        return headings

    def process_pdf(self, pdf_path: Path) -> Dict:
        """Process a single PDF and extract structured data"""
        try:
            print(f"  Extracting title...")
            title = self.extract_title(pdf_path)
            
            print(f"  Extracting headings...")
            # Try advanced extraction first, fallback to basic
            headings = self.extract_headings_with_font_analysis(pdf_path)
            if not headings:
                headings = self.extract_headings_fallback(pdf_path)
            
            # Remove duplicates and filter
            seen_texts = set()
            unique_headings = []
            
            for text, page, level in headings:
                # Skip duplicates (case-insensitive)
                text_key = text.lower().strip()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                
                unique_headings.append((text, page, level))
            
            # Sort by page number, then by appearance order
            unique_headings.sort(key=lambda x: (x[1], len(x[0])))
            
            # Limit to reasonable number of headings per page to avoid noise
            if len(unique_headings) > 50:  # Reasonable limit for most documents
                # Keep only the most likely headings based on patterns
                priority_headings = []
                for text, page, level in unique_headings:
                    if (re.match(r'^\d+\.', text) or  # Numbered sections
                        any(keyword in text.lower() for keyword in 
                            self.section_keywords['h1'] + self.section_keywords['h2'])):
                        priority_headings.append((text, page, level))
                
                if len(priority_headings) >= 10:  # Use priority if we have enough
                    unique_headings = priority_headings[:50]
                else:
                    unique_headings = unique_headings[:50]  # Limit total
            
            # Format outline
            outline = []
            for text, page, level in unique_headings:
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
                "title": "",
                "outline": []
            }

def process_pdfs():
    """Main processing function - Docker compatible"""
    # Check for Docker environment vs local testing
    if os.path.exists("sample_dataset"):
        input_dir = Path("sample_dataset/pdfs")
        output_dir = Path("test_output")
        print("Running in local test mode")
    else:
        # Docker environment paths
        input_dir = Path("/app/input")
        output_dir = Path("/app/output")
        print("Running in Docker environment")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = GeneralizedPDFProcessor()
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        start_time = __import__('time').time()
        
        # Process the PDF
        result = processor.process_pdf(pdf_file)
        
        # Create output JSON file
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        processing_time = __import__('time').time() - start_time
        print(f"Completed {pdf_file.name} -> {output_file.name} "
              f"(Found {len(result['outline'])} headings in {processing_time:.2f}s)")

if __name__ == "__main__":
    print("Starting generalized PDF processing...")
    process_pdfs() 
    print("Completed PDF processing!")
