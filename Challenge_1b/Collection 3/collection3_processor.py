#!/usr/bin/env python3
"""
🏆 Adobe India Hackathon 2025 - Challenge 1B Optimized Processor
Persona-Driven Document Intelligence - Generalizable Solution

🎯 REQUIREMENTS COMPLIANCE:
- Fully offline, CPU-only processing
- No hardcoding, domain-agnostic
- Semantic similarity matching
- Max 60 seconds execution time
- Handles persona.txt and job.txt inputs
"""

import fitz  # PyMuPDF
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import argparse
import time
import datetime
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data (offline after first run)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


@dataclass
class DocumentSection:
    """Represents an extracted document section"""
    document: str
    section_title: str
    content: str
    page_number: int
    importance_score: float
    relevance_score: float
    semantic_score: float
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    
    @property
    def combined_score(self) -> float:
        """Calculate weighted combined score"""
        return (0.4 * self.semantic_score + 
                0.3 * self.importance_score + 
                0.3 * self.relevance_score)


@dataclass
class PersonaContext:
    """Context extracted from persona and job descriptions"""
    persona_text: str
    job_text: str
    keywords: List[str]
    priorities: List[str]
    domain_indicators: List[str]


class OptimizedDocumentProcessor:
    """Generalizable, semantic-based document processor"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        
        if self.debug:
            print("✅ Optimized Document Processor initialized")
    
    def process_documents(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        Main processing pipeline - fully generalizable
        
        Args:
            input_path: Path to input directory containing PDFs, persona.txt, job.txt
            output_path: Path for output.json
            
        Returns:
            dict: Structured results
        """
        start_time = time.time()
        
        if self.debug:
            print(f"🔍 Processing documents from: {input_path}")
        
        # Load persona and job context
        persona_context = self._load_persona_context(input_path)
        if not persona_context:
            return {"error": "Could not load persona or job context"}
        
        # Extract all document sections
        all_sections = self._extract_all_sections(input_path, persona_context)
        
        if not all_sections:
            return {"error": "No sections extracted from documents"}
        
        # Perform semantic analysis and ranking
        ranked_sections = self._semantic_ranking(all_sections, persona_context)
        
        # Generate final output
        output = self._generate_output(ranked_sections, persona_context, input_path)
        
        # Save output
        self._save_output(output, output_path)
        
        execution_time = time.time() - start_time
        if self.debug:
            print(f"⏱️ Total execution time: {execution_time:.2f} seconds")
        
        return output
    
    def _load_persona_context(self, input_path: Path) -> Optional[PersonaContext]:
        """Load and parse persona.txt and job.txt"""
        try:
            persona_file = input_path / "persona.txt"
            job_file = input_path / "job.txt"
            
            # Fallback to JSON format if txt files don't exist
            if not persona_file.exists():
                persona_file = input_path / "challenge1b_input.json"
                if persona_file.exists():
                    return self._load_json_context(persona_file)
            
            if not persona_file.exists() or not job_file.exists():
                if self.debug:
                    print(f"❌ Missing context files: {persona_file}, {job_file}")
                return None
            
            persona_text = persona_file.read_text(encoding='utf-8').strip()
            job_text = job_file.read_text(encoding='utf-8').strip()
            
            # Extract semantic keywords and priorities
            keywords = self._extract_keywords(persona_text + " " + job_text)
            priorities = self._extract_priorities(job_text)
            domain_indicators = self._detect_domain(persona_text + " " + job_text)
            
            return PersonaContext(
                persona_text=persona_text,
                job_text=job_text,
                keywords=keywords,
                priorities=priorities,
                domain_indicators=domain_indicators
            )
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error loading context: {e}")
            return None
    
    def _load_json_context(self, json_file: Path) -> Optional[PersonaContext]:
        """Fallback: Load context from JSON format"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            persona = data.get("persona", {})
            job_data = data.get("job_to_be_done", {})
            
            persona_text = f"{persona.get('role', '')} {persona.get('description', '')}"
            job_text = str(job_data.get('task', '')) if isinstance(job_data, dict) else str(job_data)
            
            keywords = self._extract_keywords(persona_text + " " + job_text)
            priorities = self._extract_priorities(job_text)
            domain_indicators = self._detect_domain(persona_text + " " + job_text)
            
            return PersonaContext(
                persona_text=persona_text,
                job_text=job_text,
                keywords=keywords,
                priorities=priorities,
                domain_indicators=domain_indicators
            )
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error loading JSON context: {e}")
            return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords using TF-IDF and linguistic analysis"""
        # Clean and tokenize
        clean_text = self._clean_text(text)
        words = word_tokenize(clean_text.lower())
        
        # Filter meaningful words
        keywords = []
        for word in words:
            if (len(word) > 2 and 
                word not in self.stop_words and 
                word.isalpha() and
                not word.isdigit()):
                keywords.append(self.stemmer.stem(word))
        
        # Use frequency analysis for top keywords
        word_freq = Counter(keywords)
        top_keywords = [word for word, freq in word_freq.most_common(20)]
        
        return top_keywords
    
    def _extract_priorities(self, job_text: str) -> List[str]:
        """Extract priority indicators from job description"""
        priority_patterns = [
            r'(?:must|need|require|important|critical|essential|priority)\s+(\w+(?:\s+\w+){0,3})',
            r'(?:focus|emphasize|prioritize)\s+(?:on\s+)?(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,2})\s+(?:is|are)\s+(?:crucial|vital|key|important)'
        ]
        
        priorities = []
        for pattern in priority_patterns:
            matches = re.findall(pattern, job_text.lower())
            priorities.extend(matches)
        
        return list(set(priorities))[:10]  # Top 10 priorities
    
    def _detect_domain(self, text: str) -> List[str]:
        """Detect domain indicators from context"""
        domain_keywords = {
            'food': ['food', 'recipe', 'cooking', 'ingredient', 'meal', 'dish', 'cuisine'],
            'finance': ['financial', 'investment', 'budget', 'cost', 'revenue', 'profit'],
            'travel': ['travel', 'trip', 'destination', 'hotel', 'flight', 'itinerary'],
            'hr': ['employee', 'hiring', 'recruitment', 'workplace', 'staff', 'human resources'],
            'education': ['learning', 'education', 'student', 'course', 'curriculum', 'academic'],
            'healthcare': ['health', 'medical', 'patient', 'treatment', 'clinical', 'hospital'],
            'technology': ['software', 'system', 'technical', 'development', 'programming'],
            'business': ['business', 'corporate', 'company', 'organization', 'management']
        }
        
        text_lower = text.lower()
        detected_domains = []
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score >= 2:  # At least 2 domain keywords present
                detected_domains.append(domain)
        
        return detected_domains
    
    def _extract_all_sections(self, input_path: Path, context: PersonaContext) -> List[DocumentSection]:
        """Extract sections from all PDF documents"""
        all_sections = []
        
        # Find PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            # Check in subdirectories
            for subdir in input_path.iterdir():
                if subdir.is_dir():
                    pdf_files.extend(subdir.glob("*.pdf"))
        
        if self.debug:
            print(f"📄 Found {len(pdf_files)} PDF documents")
        
        for pdf_file in pdf_files:
            sections = self._extract_document_sections(pdf_file, context)
            all_sections.extend(sections)
            
            if self.debug:
                print(f"📖 Extracted {len(sections)} sections from {pdf_file.name}")
        
        return all_sections
    
    def _extract_document_sections(self, pdf_path: Path, context: PersonaContext) -> List[DocumentSection]:
        """Extract sections from a single PDF document"""
        sections = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks with structure
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    # Extract and clean text
                    block_text = self._extract_block_text(block)
                    if not block_text or len(block_text.strip()) < 10:
                        continue
                    
                    # Split into meaningful segments
                    segments = self._segment_text(block_text)
                    
                    for segment in segments:
                        if len(segment.strip()) < 10:
                            continue
                        
                        # Calculate semantic scores
                        semantic_score = self._calculate_semantic_similarity(segment, context)
                        importance_score = self._calculate_importance_score(segment, context)
                        relevance_score = self._calculate_relevance_score(segment, context)
                        
                        # Filter low-quality sections
                        if semantic_score < 0.1 and importance_score < 0.1:
                            continue
                        
                        # Generate meaningful title
                        section_title = self._generate_semantic_title(segment, context)
                        
                        section = DocumentSection(
                            document=pdf_path.name,
                            section_title=section_title,
                            content=segment,
                            page_number=page_num + 1,
                            importance_score=importance_score,
                            relevance_score=relevance_score,
                            semantic_score=semantic_score,
                            start_pos=(block["bbox"][0], block["bbox"][1]),
                            end_pos=(block["bbox"][2], block["bbox"][3])
                        )
                        
                        sections.append(section)
            
            doc.close()
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error processing {pdf_path}: {e}")
        
        return sections
    
    def _extract_block_text(self, block: Dict) -> str:
        """Extract clean text from a PDF block"""
        block_text = ""
        for line in block["lines"]:
            for span in line["spans"]:
                block_text += span["text"] + " "
        
        return self._clean_text(block_text)
    
    def _segment_text(self, text: str) -> List[str]:
        """Intelligently segment text into meaningful parts"""
        segments = []
        
        # Split by sentences first
        sentences = sent_tokenize(text)
        
        # Group sentences into meaningful segments
        current_segment = ""
        for sentence in sentences:
            if len(current_segment + sentence) < 500:  # Max segment length
                current_segment += sentence + " "
            else:
                if current_segment.strip():
                    segments.append(current_segment.strip())
                current_segment = sentence + " "
        
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        # Also include the full text as a segment
        if len(text) < 1000:
            segments.append(text)
        
        # Split by common delimiters for variety
        for delimiter in ['\n\n', '•', '○', '1.', '2.', '3.']:
            parts = text.split(delimiter)
            for part in parts:
                part = part.strip()
                if 50 <= len(part) <= 500:  # Reasonable length
                    segments.append(part)
        
        return list(set(segments))  # Remove duplicates
    
    def _calculate_semantic_similarity(self, text: str, context: PersonaContext) -> float:
        """Calculate semantic similarity using TF-IDF and cosine similarity"""
        try:
            # Combine persona and job text for comparison
            reference_text = f"{context.persona_text} {context.job_text}"
            
            # Vectorize texts
            texts = [text, reference_text]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception:
            # Fallback to keyword matching
            return self._keyword_similarity(text, context)
    
    def _keyword_similarity(self, text: str, context: PersonaContext) -> float:
        """Fallback keyword-based similarity calculation"""
        text_lower = text.lower()
        
        # Check for persona keywords
        persona_matches = sum(1 for keyword in context.keywords if keyword in text_lower)
        persona_score = min(persona_matches / max(len(context.keywords), 1), 1.0)
        
        # Check for priority terms
        priority_matches = sum(1 for priority in context.priorities if priority in text_lower)
        priority_score = min(priority_matches / max(len(context.priorities), 1), 1.0)
        
        # Check for domain relevance
        domain_score = 0.0
        for domain in context.domain_indicators:
            if domain in text_lower:
                domain_score += 0.2
        
        return min((persona_score * 0.4 + priority_score * 0.4 + domain_score * 0.2), 1.0)
    
    def _calculate_importance_score(self, text: str, context: PersonaContext) -> float:
        """Calculate importance based on content structure and keywords"""
        score = 0.0
        text_lower = text.lower()
        
        # Structural importance indicators
        if ':' in text and len(text) < 100:  # Likely a header
            score += 0.3
        
        if re.search(r'^\s*\d+\.', text.strip()):  # Numbered list item
            score += 0.2
        
        if re.search(r'[•○▪▫]', text):  # Bullet points
            score += 0.15
        
        # Content density
        word_count = len(text.split())
        if 10 <= word_count <= 100:  # Optimal length
            score += 0.2
        elif word_count > 100:
            score += 0.1
        
        # Keyword density
        keyword_matches = sum(1 for keyword in context.keywords if keyword in text_lower)
        keyword_density = keyword_matches / max(len(text.split()), 1)
        score += min(keyword_density * 2, 0.3)
        
        return min(score, 1.0)
    
    def _calculate_relevance_score(self, text: str, context: PersonaContext) -> float:
        """Calculate relevance to the specific job/task"""
        text_lower = text.lower()
        job_lower = context.job_text.lower()
        
        # Direct job description overlap
        job_words = set(word_tokenize(job_lower))
        text_words = set(word_tokenize(text_lower))
        
        overlap = len(job_words.intersection(text_words))
        overlap_score = min(overlap / max(len(job_words), 1), 1.0)
        
        # Priority term matching
        priority_score = 0.0
        for priority in context.priorities:
            if priority in text_lower:
                priority_score += 0.2
        
        return min(overlap_score * 0.6 + priority_score * 0.4, 1.0)
    
    def _generate_semantic_title(self, text: str, context: PersonaContext) -> str:
        """Generate meaningful section titles based on content"""
        text = text.strip()
        
        # Look for existing headers (lines ending with colons, short lines, etc.)
        lines = text.split('\n')
        first_line = lines[0].strip()
        
        # Check if first line is a natural header
        if (len(first_line) < 60 and 
            (first_line.endswith(':') or 
             first_line.isupper() or 
             re.match(r'^[A-Z][a-z\s]+$', first_line))):
            return first_line
        
        # Extract key phrases using domain context
        sentences = sent_tokenize(text)
        if sentences:
            first_sentence = sentences[0]
            
            # Look for important phrases in first sentence
            for keyword in context.keywords:
                if keyword in first_sentence.lower():
                    # Extract phrase around the keyword
                    words = first_sentence.split()
                    for i, word in enumerate(words):
                        if keyword in word.lower():
                            start = max(0, i-2)
                            end = min(len(words), i+3)
                            phrase = ' '.join(words[start:end])
                            if len(phrase) < 50:
                                return phrase.strip('.,!?')
        
        # Extract meaningful phrases
        patterns = [
            r'(?:how to|steps to|guide to)\s+(.{10,40})',
            r'(.{10,40})\s+(?:process|procedure|method|approach)',
            r'(.{10,40})\s+(?:requirements|specifications|details)',
            r'(.{10,40})\s+(?:analysis|review|evaluation)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip().title()
        
        # Fallback: use first few meaningful words
        words = text.split()[:8]
        clean_words = [word.strip('.,!?()[]') for word in words if len(word) > 2]
        
        if clean_words:
            title = ' '.join(clean_words[:6])
            return title if len(title) < 60 else title[:57] + "..."
        
        return "Document Section"
    
    def _semantic_ranking(self, sections: List[DocumentSection], context: PersonaContext) -> List[DocumentSection]:
        """Rank sections using semantic analysis"""
        
        # Calculate combined scores and rank
        for section in sections:
            # Additional context-aware scoring
            section.semantic_score = self._enhanced_semantic_score(section, context)
        
        # Sort by combined score
        ranked_sections = sorted(sections, key=lambda s: s.combined_score, reverse=True)
        
        # Ensure diversity in final ranking
        final_sections = self._ensure_diversity(ranked_sections)
        
        if self.debug:
            print(f"📊 Ranked {len(final_sections)} sections by semantic relevance")
            for i, section in enumerate(final_sections[:10]):
                print(f"   {i+1}. '{section.section_title}' (Score: {section.combined_score:.3f})")
        
        return final_sections
    
    def _enhanced_semantic_score(self, section: DocumentSection, context: PersonaContext) -> float:
        """Enhanced semantic scoring with context awareness"""
        base_score = section.semantic_score
        
        # Boost for domain-specific content
        content_lower = section.content.lower()
        
        # Domain relevance boost
        for domain in context.domain_indicators:
            if domain in content_lower:
                base_score += 0.1
        
        # Job-specific terms boost
        job_words = context.job_text.lower().split()
        matches = sum(1 for word in job_words if word in content_lower)
        job_relevance = min(matches / max(len(job_words), 1), 0.3)
        base_score += job_relevance
        
        # Priority terms boost
        for priority in context.priorities:
            if priority in content_lower:
                base_score += 0.05
        
        # Length penalty for very short or very long content
        content_length = len(section.content)
        if content_length < 20:
            base_score *= 0.8
        elif content_length > 1000:
            base_score *= 0.9
        
        return min(base_score, 1.0)
    
    def _ensure_diversity(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Ensure diversity in selected sections"""
        final_sections = []
        seen_titles = set()
        documents_count = defaultdict(int)
        max_per_doc = 3  # Max sections per document
        
        for section in sections:
            # Avoid exact duplicate titles
            title_key = section.section_title.lower().strip()
            
            # Skip if too many from same document
            if documents_count[section.document] >= max_per_doc:
                continue
            
            # Skip exact duplicates, but allow similar content
            if title_key not in seen_titles:
                final_sections.append(section)
                seen_titles.add(title_key)
                documents_count[section.document] += 1
                
                if len(final_sections) >= 20:  # Reasonable limit
                    break
        
        return final_sections
    
    def _generate_output(self, sections: List[DocumentSection], context: PersonaContext, input_path: Path) -> Dict[str, Any]:
        """Generate final structured output"""
        
        # Get input documents list
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            for subdir in input_path.iterdir():
                if subdir.is_dir():
                    pdf_files.extend(subdir.glob("*.pdf"))
        
        input_documents = [pdf.name for pdf in pdf_files]
        
        # Format main sections (top ranked)
        main_sections = sections[:5]
        sub_sections = sections[5:15] if len(sections) > 5 else []
        
        output = {
            "metadata": {
                "input_documents": input_documents,
                "persona": context.persona_text[:200] + "..." if len(context.persona_text) > 200 else context.persona_text,
                "job_to_be_done": context.job_text[:200] + "..." if len(context.job_text) > 200 else context.job_text,
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "total_sections_analyzed": len(sections),
                "semantic_domains": context.domain_indicators
            },
            "extracted_sections": [
                {
                    "document": section.document,
                    "section_title": section.section_title,
                    "importance_rank": i + 1,
                    "page_number": section.page_number,
                    "semantic_score": round(section.semantic_score, 3),
                    "combined_score": round(section.combined_score, 3)
                }
                for i, section in enumerate(main_sections)
            ],
            "subsection_analysis": [
                {
                    "document": section.document,
                    "refined_text": section.content[:200] + "..." if len(section.content) > 200 else section.content,
                    "page_number": section.page_number,
                    "relevance_score": round(section.relevance_score, 3)
                }
                for section in sub_sections
            ]
        }
        
        return output
    
    def _save_output(self, output: Dict[str, Any], output_path: Path) -> None:
        """Save output to JSON file"""
        output_file = output_path / "output.json"
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        if self.debug:
            print(f"💾 Output saved to: {output_file}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Optimized Document Intelligence Processor")
    parser.add_argument("--input", default="/app/input", help="Input directory path")
    parser.add_argument("--output", default="/app/output", help="Output directory path")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    processor = OptimizedDocumentProcessor(debug=args.debug)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        return 1
    
    try:
        result = processor.process_documents(input_path, output_path)
        
        if "error" in result:
            print(f"❌ ERROR: {result['error']}")
            return 1
        
        if args.debug:
            print("✅ Processing completed successfully")
            sections_count = len(result.get("extracted_sections", []))
            subsections_count = len(result.get("subsection_analysis", []))
            print(f"📊 Generated {sections_count} main sections, {subsections_count} subsections")
        
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
