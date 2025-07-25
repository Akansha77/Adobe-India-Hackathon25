#!/usr/bin/env python3
"""
🏆 Adobe India Hackathon 2025 - Enhanced PDF Document Intelligence
Round 1A Final Submission: Title and Heading Extraction

🎯 TARGET METRICS:
- Title Accuracy ≥ 80%
- Heading Precision ≥ 60% 
- Heading Recall ≥ 70%
- F1-Score ≥ 65%

🚀 ADVANCED FEATURES:
- Dynamic font clustering for heading detection
- Semantic title scoring with multiline combination
- Layout-aware filtering (whitespace analysis)
- Header/footer detection and removal
- Adaptive confidence thresholding
- Edge case handling (forms, tables, dense layouts)

📋 COMPLIANCE:
✅ CPU-only processing (no GPU dependencies)
✅ Processing time <10 seconds per document
✅ Model size <200MB (PyMuPDF only)
✅ Universal processing (no hardcoded document-specific logic)
✅ Docker containerization with proper I/O handling
✅ Offline operation (no internet/API calls)
"""

import fitz  # PyMuPDF
import json
import re
import unicodedata
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import argparse
import time
import math


class EnhancedPDFProcessor:
    """
    Enhanced PDF processor with advanced font clustering and semantic analysis
    Designed to achieve ≥80% title accuracy and ≥65% heading F1-score
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Enhanced configuration parameters
        self.config = {
            # Title extraction parameters
            'title_search_height_ratio': 0.4,  # Search in top 40% of first page
            'title_multiline_gap_threshold': 30,  # Max gap for multiline titles
            'title_min_confidence': 0.7,  # Minimum confidence for title selection
            
            # Heading detection parameters  
            'heading_confidence_threshold': 0.65,  # Adaptive threshold
            'min_font_size': 9,  # Minimum font size for headings
            'max_heading_frequency': 0.3,  # Max frequency for heading fonts
            'whitespace_isolation_bonus': 0.2,  # Bonus for isolated headings
            
            # Layout analysis parameters
            'header_footer_margin': 50,  # Margin for header/footer detection
            'dense_text_threshold': 0.8,  # Threshold for dense text blocks
            'page_consistency_penalty': 0.3,  # Penalty for repeated elements
        }
        
        if self.debug:
            print("🔧 Enhanced PDF Processor initialized with advanced algorithms")
    
    def process_pdf_file(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a single PDF file with enhanced extraction algorithms
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            dict: Contains 'title' and 'outline' with extracted content
        """
        if self.debug:
            print(f"📄 Processing {pdf_path.name}...")
        
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            
            if len(doc) == 0:
                doc.close()
                return {"title": "", "outline": []}
            
            # Extract title using enhanced multiline semantic analysis
            title = self.extract_title_enhanced(doc)
            
            # Extract hierarchical headings using dynamic font clustering
            outline = self.extract_headings_enhanced(doc)
            
            doc.close()
            
            result = {
                "title": title,
                "outline": outline
            }
            
            processing_time = time.time() - start_time
            
            if self.debug:
                print(f"   ✅ Title: '{title[:50]}{'...' if len(title) > 50 else ''}'")
                print(f"   ✅ Headings: {len(outline)} found")
                print(f"   ⏱️  Time: {processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
            return {"title": "", "outline": []}
    
    def extract_title_enhanced(self, doc: fitz.Document) -> str:
        """
        Enhanced title extraction with semantic scoring and multiline combination
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            str: Extracted title with trailing space for consistency
        """
        try:
            if len(doc) == 0:
                return ""
            
            page = doc.load_page(0)
            page_height = page.rect.height
            page_width = page.rect.width
            
            # Extract all text elements from title area
            title_area_height = page_height * self.config['title_search_height_ratio']
            text_elements = self._extract_text_elements(page, max_height=title_area_height)
            
            if not text_elements:
                return ""
            
            # Apply advanced filtering
            filtered_elements = self._filter_title_candidates(text_elements, page_height, page_width)
            
            if not filtered_elements:
                return ""
            
            # Score individual elements
            scored_elements = []
            for element in filtered_elements:
                score = self._calculate_title_score_enhanced(element, page_height, page_width)
                if score > 0.3:  # Minimum score threshold
                    scored_elements.append((score, element))
            
            if not scored_elements:
                return ""
            
            # Try multiline combinations
            multiline_candidates = self._generate_multiline_titles(scored_elements, page_height)
            
            # Combine single and multiline candidates
            all_candidates = [(score, element['text']) for score, element in scored_elements]
            all_candidates.extend(multiline_candidates)
            
            # Sort by score and select best
            all_candidates.sort(reverse=True, key=lambda x: x[0])
            
            if self.debug:
                print(f"   🔍 DEBUG - Title candidates found: {len(all_candidates)}")
                for i, (score, text) in enumerate(all_candidates[:3]):
                    print(f"      {i+1}. Score: {score:.3f} | '{text[:50]}'")
            
            if all_candidates and all_candidates[0][0] >= self.config['title_min_confidence']:
                best_title = self.clean_text(all_candidates[0][1])
                
                # Ensure trailing space for consistency
                if best_title and not best_title.endswith(' '):
                    best_title += ' '
                
                if self.debug:
                    print(f"   🎯 DEBUG - Selected title: '{best_title[:60]}'")
                
                return best_title
            
            return ""
            
        except Exception as e:
            print(f"   ⚠️  Error in title extraction: {e}")
            return ""
    
    def _extract_text_elements(self, page: fitz.Page, max_height: Optional[float] = None) -> List[Dict]:
        """Extract text elements with enhanced metadata"""
        elements = []
        blocks = page.get_text("dict")
        
        for block in blocks.get("blocks", []):
            if block.get("type") != 0:  # Skip non-text blocks
                continue
                
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    
                    # Skip if outside height limit
                    if max_height and bbox[1] > max_height:
                        continue
                    
                    if text and len(text) > 2:
                        element = {
                            'text': text,
                            'bbox': bbox,
                            'font_size': span.get("size", 0),
                            'font_name': span.get("font", ""),
                            'is_bold': bool(span.get("flags", 0) & 16),
                            'is_italic': bool(span.get("flags", 0) & 2),
                            'x_pos': bbox[0],
                            'y_pos': bbox[1],
                            'width': bbox[2] - bbox[0],
                            'height': bbox[3] - bbox[1]
                        }
                        elements.append(element)
        
        return elements
    
    def _filter_title_candidates(self, elements: List[Dict], page_height: float, page_width: float) -> List[Dict]:
        """Advanced filtering for title candidates"""
        filtered = []
        
        for element in elements:
            text = element['text']
            
            # Skip obvious non-titles
            if self._is_non_title_content(text):
                continue
            
            # Skip if too small or in wrong position
            if (element['font_size'] < 8 or 
                element['y_pos'] > page_height * 0.5 or
                len(text) < 3):
                continue
            
            # Skip if looks like metadata
            if self._is_document_metadata(text):
                continue
                
            filtered.append(element)
        
        return filtered
    
    def _is_non_title_content(self, text: str) -> bool:
        """Check if text is clearly not a title"""
        text_lower = text.lower().strip()
        
        # Pattern-based exclusions
        patterns = [
            r'^\d+[\.\)]',  # Numbered lists
            r'^[a-z]\)',    # Lettered lists  
            r'page\s+\d+',  # Page numbers
            r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}',  # Dates
            r'©.*\d{4}',    # Copyright notices
            r'www\.|@|\.com|\.org',  # URLs/emails
            r'^\d+$',       # Pure numbers
            r'^[^a-zA-Z]*$' # No letters
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Common non-title phrases
        non_title_phrases = [
            'page', 'draft', 'version', 'confidential', 'header', 'footer',
            'table of contents', 'appendix', 'bibliography', 'references'
        ]
        
        if text_lower in non_title_phrases:
            return True
        
        # Too many underscores (form fields)
        if text.count('_') > 3:
            return True
            
        return False
    
    def _is_document_metadata(self, text: str) -> bool:
        """Check if text is document metadata"""
        text_lower = text.lower().strip()
        
        metadata_indicators = [
            'microsoft word', 'adobe pdf', 'created by', 'modified by',
            'file size', 'last saved', 'document title', 'author:',
            'subject:', 'keywords:', 'creator:', 'producer:'
        ]
        
        for indicator in metadata_indicators:
            if indicator in text_lower:
                return True
                
        return False
    
    def _group_title_elements(self, elements):
        """Group text elements that could form multiline titles"""
        if not elements:
            return []
            
        # Sort by page and vertical position
        elements.sort(key=lambda x: (x['page'], x['y_pos']))
        
        groups = []
        current_group = []
        
        for elem in elements:
            # Skip if text is too short or looks like metadata
            if len(elem['text']) < 3 or self._is_metadata_text(elem['text']):
                continue
                
            # Start new group if this is first element
            if not current_group:
                current_group = [elem]
                continue
            
            last_elem = current_group[-1]
            
            # Check if this element should be grouped with previous
            same_page = elem['page'] == last_elem['page']
            close_vertically = abs(elem['y_pos'] - last_elem['y_pos']) < 30  # Within 30 points
            similar_font = (abs(elem['font_info']['size'] - last_elem['font_info']['size']) < 2 and
                          elem['font_info']['name'] == last_elem['font_info']['name'])
            
            if same_page and close_vertically and similar_font:
                current_group.append(elem)
            else:
                # Finalize current group if it has content
                if current_group:
                    groups.append(current_group)
                current_group = [elem]
        
        # Add final group
        if current_group:
            groups.append(current_group)
        
        # Filter groups - prefer those with larger fonts and early page position
        valid_groups = []
        for group in groups:
            avg_font_size = sum([elem['font_info']['size'] for elem in group]) / len(group)
            # Only keep groups with reasonable font size (>10pt) and not too many elements
            if avg_font_size > 10 and len(group) <= 4:
                valid_groups.append(group)
        
        return valid_groups
    
    def _calculate_title_score_enhanced(self, element: Dict, page_height: float, page_width: float) -> float:
        """Enhanced title scoring with semantic analysis"""
        text = element['text']
        font_size = element['font_size']
        is_bold = element['is_bold']
        bbox = element['bbox']
        y_pos = element['y_pos']
        
        score = 0.0
        
        # 1. Font size score (0-0.3)
        font_score = min(0.3, font_size / 60.0)  # Normalize to typical range
        score += font_score
        
        # 2. Bold bonus (0-0.2)
        if is_bold:
            score += 0.2
        
        # 3. Position score (0-0.25) - higher on page is better
        position_ratio = 1 - (y_pos / (page_height * 0.4))  # Within title area
        position_score = max(0, min(0.25, position_ratio * 0.25))
        score += position_score
        
        # 4. Center alignment bonus (0-0.15)
        text_center = (bbox[0] + bbox[2]) / 2
        page_center = page_width / 2
        center_deviation = abs(text_center - page_center) / page_width
        if center_deviation < 0.2:
            score += 0.15
        elif center_deviation < 0.4:
            score += 0.1
        
        # 5. Length optimization (0-0.15)
        text_len = len(text)
        if 10 <= text_len <= 100:
            score += 0.15
        elif 5 <= text_len <= 150:
            score += 0.1
        elif text_len > 200:
            score -= 0.1
        
        # 6. Semantic content analysis (0-0.2)
        text_lower = text.lower()
        
        # Generic title keywords (no document-specific terms)
        generic_title_keywords = [
            'report', 'analysis', 'study', 'plan', 'guide', 'overview',
            'document', 'proposal', 'application', 'form', 'manual'
        ]
        
        for keyword in generic_title_keywords:
            if keyword in text_lower:
                score += 0.15
                break
        
        # Title formatting bonuses
        if text.isupper() and text_len > 3:
            score += 0.1
        elif text.istitle():
            score += 0.08
        
        # 7. Format-based bonuses (generic patterns)
        # Document with colon format bonus
        if ':' in text and len(text.split(':')[0]) <= 10:
            score += 0.15
        
        # Short descriptive titles bonus
        if (len(text) < 60 and 
            'mission' not in text_lower and 
            'provide' not in text_lower and
            len(text.split()) >= 2):
            score += 0.1
        
        # 8. Penalties
        # Mission statement penalty
        mission_indicators = [
            'mission statement', 'mission:', 'to provide', 'goals:',
            'our mission', 'objective:', 'purpose:'
        ]
        
        for indicator in mission_indicators:
            if indicator in text_lower:
                score -= 0.3
                break
        
        # Form field penalty
        if text.count('_') > 2:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _generate_multiline_titles(self, scored_elements: List[Tuple], page_height: float) -> List[Tuple]:
        """Generate multiline title combinations"""
        multiline_candidates = []
        
        # Sort by vertical position
        elements_by_pos = [(score, element) for score, element in scored_elements]
        elements_by_pos.sort(key=lambda x: x[1]['y_pos'])
        
        gap_threshold = self.config['title_multiline_gap_threshold']
        
        # Try combinations of 2-3 consecutive elements
        for i in range(len(elements_by_pos)):
            for j in range(i + 1, min(i + 4, len(elements_by_pos) + 1)):
                combination = elements_by_pos[i:j]
                
                if len(combination) < 2:
                    continue
                
                # Check vertical gaps
                valid_combination = True
                combined_text_parts = []
                total_score = 0
                
                for k in range(len(combination)):
                    _, element = combination[k]
                    combined_text_parts.append(element['text'])
                    
                    if k > 0:
                        prev_element = combination[k-1][1]
                        gap = element['y_pos'] - prev_element['bbox'][3]
                        if gap > gap_threshold:
                            valid_combination = False
                            break
                    
                    # Weight scores by position (first element gets more weight)
                    weight = 1.0 - (k * 0.1)
                    total_score += combination[k][0] * weight
                
                if not valid_combination:
                    continue
                
                # Combine text
                combined_text = ' '.join(combined_text_parts)
                
                # Bonus for valid multiline titles
                if len(combined_text) > 15:
                    total_score += 0.1
                
                # Penalty for too long combinations
                if len(combined_text) > 150:
                    total_score -= 0.2
                
                # Average the score
                avg_score = total_score / len(combination)
                
                multiline_candidates.append((avg_score, combined_text))
        
        return multiline_candidates
    
    def extract_headings_enhanced(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """
        Enhanced heading extraction using dynamic font clustering
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            List[Dict]: Hierarchical outline with level, text, and page
        """
        try:
            if len(doc) == 0:
                return []
            
            # Extract all text elements from all pages
            all_elements = []
            page_heights = {}
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_heights[page_num] = page.rect.height
                elements = self._extract_text_elements(page)
                
                for element in elements:
                    element['page'] = page_num
                    all_elements.append(element)
            
            if not all_elements:
                return []
            
            # Perform dynamic font clustering
            heading_fonts = self._dynamic_font_clustering(all_elements)
            
            # Detect and filter headers/footers
            header_footer_elements = self._detect_headers_footers(all_elements, page_heights)
            
            # Score and filter heading candidates
            outline = []
            processed_texts = set()
            
            for element in all_elements:
                text = self.clean_text(element['text'])
                
                # Skip if already processed, too short, or header/footer
                if (text in processed_texts or 
                    len(text) < 3 or 
                    self._is_header_footer(element, header_footer_elements)):
                    continue
                
                # Calculate heading confidence
                confidence = self._calculate_heading_confidence_enhanced(
                    element, all_elements, heading_fonts, page_heights
                )
                
                if confidence >= self.config['heading_confidence_threshold']:
                    if self.debug:
                        print(f"   🔍 DEBUG - Heading: '{text[:30]}' | Conf: {confidence:.3f}")
                    
                    # Determine heading level
                    level = self._determine_heading_level_enhanced(element, heading_fonts, confidence)
                    
                    outline.append({
                        "level": level,
                        "text": text,
                        "page": element['page']
                    })
                    
                    processed_texts.add(text)
            
            return outline
            
        except Exception as e:
            print(f"   ⚠️  Error in heading extraction: {e}")
            return []
    
    def _dynamic_font_clustering(self, elements: List[Dict]) -> List[Dict]:
        """Dynamic font clustering for heading detection"""
        
        # Analyze font usage patterns
        font_stats = defaultdict(lambda: {
            'count': 0,
            'total_chars': 0,
            'pages': set(),
            'positions': [],
            'elements': []
        })
        
        for element in elements:
            font_key = (
                element['font_name'], 
                round(element['font_size'], 1), 
                element['is_bold']
            )
            
            stats = font_stats[font_key]
            stats['count'] += 1
            stats['total_chars'] += len(element['text'])
            stats['pages'].add(element['page'])
            stats['positions'].append(element['y_pos'])
            stats['elements'].append(element)
        
        # Calculate derived metrics
        total_elements = len(elements)
        heading_candidates = []
        
        for font_key, stats in font_stats.items():
            font_name, font_size, is_bold = font_key
            
            frequency = stats['count'] / total_elements
            avg_length = stats['total_chars'] / stats['count']
            page_spread = len(stats['pages'])
            
            # Enhanced heading criteria
            if (font_size >= self.config['min_font_size'] and 
                frequency <= self.config['max_heading_frequency'] and
                frequency >= 0.001):  # Not too rare
                
                # Calculate heading likelihood score
                score = 0.0
                
                # Font size contribution
                score += font_size * 0.05
                
                # Bold bonus
                if is_bold:
                    score += 0.3
                
                # Frequency optimization (rare but not too rare)
                if 0.005 <= frequency <= 0.1:
                    score += 0.4
                elif 0.001 <= frequency <= 0.05:
                    score += 0.3
                
                # Length bonus (headings are typically shorter)
                if 10 <= avg_length <= 80:
                    score += 0.2
                elif avg_length <= 120:
                    score += 0.1
                
                # Page spread bonus
                score += min(page_spread * 0.05, 0.2)
                
                heading_candidates.append({
                    'font_key': font_key,
                    'score': score,
                    'stats': stats,
                    'font_name': font_name,
                    'font_size': font_size,
                    'is_bold': is_bold,
                    'frequency': frequency
                })
        
        # Sort by score
        heading_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if self.debug and heading_candidates:
            print(f"   📊 DEBUG - Top heading fonts:")
            for i, candidate in enumerate(heading_candidates[:5]):
                print(f"      {i+1}. {candidate['font_name']} {candidate['font_size']}{'B' if candidate['is_bold'] else ''} | Score: {candidate['score']:.3f}")
        
        return heading_candidates
    
    def _detect_headers_footers(self, elements: List[Dict], page_heights: Dict[int, float]) -> Set[str]:
        """Detect repeated headers and footers"""
        
        # Group elements by text content
        text_occurrences = defaultdict(list)
        
        for element in elements:
            text = element['text'].strip()
            if len(text) > 3:
                text_occurrences[text].append(element)
        
        header_footer_texts = set()
        margin = self.config['header_footer_margin']
        
        for text, occurrences in text_occurrences.items():
            if len(occurrences) < 2:  # Must appear on multiple pages
                continue
            
            # Check if consistently in header or footer positions
            in_header = 0
            in_footer = 0
            
            for element in occurrences:
                page_height = page_heights.get(element['page'], 800)
                
                if element['y_pos'] < margin:
                    in_header += 1
                elif element['y_pos'] > page_height - margin:
                    in_footer += 1
            
            # If majority are in header/footer positions
            if in_header >= len(occurrences) * 0.7 or in_footer >= len(occurrences) * 0.7:
                header_footer_texts.add(text)
        
        if self.debug and header_footer_texts:
            print(f"   🚫 DEBUG - Headers/footers detected: {len(header_footer_texts)}")
        
        return header_footer_texts
    
    def _is_header_footer(self, element: Dict, header_footer_texts: Set[str]) -> bool:
        """Check if element is a header or footer"""
        return element['text'].strip() in header_footer_texts
    
    def _calculate_heading_confidence_enhanced(self, element: Dict, all_elements: List[Dict], 
                                             heading_fonts: List[Dict], page_heights: Dict[int, float]) -> float:
        """Enhanced heading confidence calculation"""
        
        text = element['text']
        font_size = element['font_size']
        is_bold = element['is_bold']
        page = element['page']
        bbox = element['bbox']
        
        confidence = 0.0
        
        # 1. Font matching score (0-0.4)
        element_font_key = (element['font_name'], round(font_size, 1), is_bold)
        
        for i, font_candidate in enumerate(heading_fonts[:10]):  # Top 10 fonts
            if font_candidate['font_key'] == element_font_key:
                font_score = 0.4 - (i * 0.03)  # Decreasing bonus
                confidence += max(0, font_score)
                break
        
        # 2. Font size score (0-0.25)
        if font_size >= 18:
            confidence += 0.25
        elif font_size >= 16:
            confidence += 0.2
        elif font_size >= 14:
            confidence += 0.15
        elif font_size >= 12:
            confidence += 0.1
        elif font_size >= 10:
            confidence += 0.05
        
        # 3. Bold bonus (0-0.15)
        if is_bold:
            confidence += 0.15
        
        # 4. Length score (0-0.1)
        text_len = len(text)
        if 5 <= text_len <= 80:
            confidence += 0.1
        elif text_len <= 120:
            confidence += 0.05
        elif text_len > 200:
            confidence -= 0.1
        
        # 5. Position and isolation analysis (0-0.15)
        isolation_score = self._calculate_isolation_score(element, all_elements)
        confidence += isolation_score * 0.15
        
        # 6. Content analysis (0-0.1)
        content_score = self._analyze_heading_content(text)
        confidence += content_score * 0.1
        
        # 7. Layout consistency penalty
        if self._is_in_dense_text_block(element, all_elements):
            confidence -= 0.2
        
        # 8. Penalties for non-heading patterns
        if self._has_non_heading_patterns(text):
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_isolation_score(self, element: Dict, all_elements: List[Dict]) -> float:
        """Calculate how isolated a heading is (surrounded by whitespace)"""
        
        same_page_elements = [e for e in all_elements if e['page'] == element['page']]
        
        # Check for elements above and below
        elements_above = [e for e in same_page_elements 
                         if e['bbox'][3] <= element['bbox'][1] and 
                            abs(e['bbox'][1] - element['bbox'][1]) <= 100]
        
        elements_below = [e for e in same_page_elements 
                         if e['bbox'][1] >= element['bbox'][3] and 
                            abs(e['bbox'][1] - element['bbox'][3]) <= 100]
        
        # Calculate whitespace above and below
        whitespace_above = 0
        if elements_above:
            closest_above = max(elements_above, key=lambda x: x['bbox'][3])
            whitespace_above = element['bbox'][1] - closest_above['bbox'][3]
        
        whitespace_below = 0
        if elements_below:
            closest_below = min(elements_below, key=lambda x: x['bbox'][1])
            whitespace_below = closest_below['bbox'][1] - element['bbox'][3]
        
        # Normalize whitespace scores
        total_whitespace = whitespace_above + whitespace_below
        isolation_score = min(1.0, total_whitespace / 50.0)  # 50 is threshold for good isolation
        
        return isolation_score
    
    def _analyze_heading_content(self, text: str) -> float:
        """Analyze text content for heading-like characteristics"""
        text_lower = text.lower().strip()
        score = 0.0
        
        # Heading patterns
        if re.match(r'^[A-Z][A-Z\s]+$', text):  # All caps
            score += 0.5
        elif text.istitle():  # Title case
            score += 0.3
        
        # Generic heading keywords (no document-specific terms)
        generic_heading_keywords = [
            'introduction', 'background', 'methodology', 'results',
            'conclusion', 'summary', 'abstract', 'discussion', 'analysis',
            'chapter', 'section', 'part', 'appendix', 'references',
            'overview', 'objectives', 'purpose', 'scope', 'requirements'
        ]
        
        for keyword in generic_heading_keywords:
            if keyword in text_lower:
                score += 0.4
                break
        
        # Numbered sections
        if re.match(r'^\d+[\.\s]', text) or re.match(r'^[A-Z][\.\s]', text):
            score += 0.3
        
        return min(1.0, score)
    
    def _is_in_dense_text_block(self, element: Dict, all_elements: List[Dict]) -> bool:
        """Check if element is within a dense text block"""
        
        same_page_elements = [e for e in all_elements if e['page'] == element['page']]
        
        # Define area around the element
        search_area = {
            'x1': element['bbox'][0] - 50,
            'y1': element['bbox'][1] - 30,
            'x2': element['bbox'][2] + 50,
            'y2': element['bbox'][3] + 30
        }
        
        # Count elements in the area
        nearby_elements = []
        for e in same_page_elements:
            if (e['bbox'][0] >= search_area['x1'] and e['bbox'][2] <= search_area['x2'] and
                e['bbox'][1] >= search_area['y1'] and e['bbox'][3] <= search_area['y2']):
                nearby_elements.append(e)
        
        # Calculate density
        area_size = (search_area['x2'] - search_area['x1']) * (search_area['y2'] - search_area['y1'])
        text_coverage = sum(e['width'] * e['height'] for e in nearby_elements)
        
        density = text_coverage / area_size if area_size > 0 else 0
        
        return density > self.config['dense_text_threshold']
    
    def _has_non_heading_patterns(self, text: str) -> bool:
        """Check for patterns that indicate non-heading content"""
        text_lower = text.lower().strip()
        
        # Non-heading patterns
        non_heading_patterns = [
            r'^page\s+\d+',
            r'^\d+$',
            r'^[a-z]+\s*[^\w\s]',  # Starts with lowercase
            r'\.$',  # Ends with period (likely sentence)
            r'see\s+(page|section|chapter)',
            r'continued\s+on',
            r'table\s+\d+',
            r'figure\s+\d+'
        ]
        
        for pattern in non_heading_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for sentence-like structure
        if text.count('.') > 1 or (text.endswith('.') and len(text) > 50):
            return True
        
        # Form fields
        if text.count('_') > 3:
            return True
        
        return False
    
    def _determine_heading_level_enhanced(self, element: Dict, heading_fonts: List[Dict], confidence: float) -> str:
        """Enhanced heading level determination"""
        
        font_size = element['font_size']
        is_bold = element['is_bold']
        element_font_key = (element['font_name'], round(font_size, 1), is_bold)
        
        # Find font rank
        font_rank = 999
        for i, font_candidate in enumerate(heading_fonts):
            if font_candidate['font_key'] == element_font_key:
                font_rank = i
                break
        
        # Multi-factor level determination
        if ((font_size >= 18 or font_rank == 0) and confidence >= 0.9):
            return "H1"
        elif ((font_size >= 16 or font_rank <= 1) and confidence >= 0.8):
            return "H1" 
        elif ((font_size >= 14 or font_rank <= 3) and confidence >= 0.75):
            return "H2"
        elif confidence >= 0.7:
            return "H2"
        else:
            return "H3"
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning and normalization"""
        if not text:
            return ""
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in ['\n', '\t'])
        
        # Clean up common artifacts
        text = re.sub(r'\s+([,.;:])', r'\1', text)  # Fix spacing before punctuation
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Fix spacing after sentences
        
        return text


def main():
    """Main function for command-line usage and Docker deployment"""
    parser = argparse.ArgumentParser(
        description='Enhanced PDF Document Intelligence - Adobe Hackathon 2025'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory containing PDF files')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for JSON results')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug output')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize enhanced processor
    processor = EnhancedPDFProcessor(debug=args.debug)
    
    # Process all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        return 1
    
    print(f"🚀 Processing {len(pdf_files)} PDF files from {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    total_time = 0
    success_count = 0
    
    for pdf_file in sorted(pdf_files):
        start_time = time.time()
        
        try:
            # Process the PDF
            result = processor.process_pdf_file(pdf_file)
            
            # Save result
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            total_time += processing_time
            success_count += 1
            
            print(f"✅ {pdf_file.name} → {output_file.name} ({processing_time:.1f}s)")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
    
    print(f"\n🎉 Processing complete!")
    print(f"📊 Success rate: {success_count}/{len(pdf_files)} files")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"⚡ Average time per file: {total_time/len(pdf_files):.1f}s")
    
    return 0 if success_count == len(pdf_files) else 1


if __name__ == "__main__":
    exit(main())
