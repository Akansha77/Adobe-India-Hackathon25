def __init__(self, debug: bool = False):
    self.debug = debug
    
    # Balanced configuration for optimal F1-score
    self.config = {
        # Title extraction
        'title_search_ratio': 0.3,
        'title_min_confidence': 0.5,
        'title_min_length': 10,
        
        # Heading detection - balanced thresholds
        'heading_confidence_threshold': 0.5,  # Moderate threshold
        'min_heading_length': 4,
        'max_heading_length': 120,
        'min_font_size': 9,
        'max_heading_frequency': 0.4,
        
        # Text combination
        'line_grouping_threshold': 8,  # pixels
        'text_merge_distance': 15,     # pixels
        'prefer_complete_phrases': True
    }

def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
    """Process a PDF file and extract title and headings"""
    try:
        if self.debug:
            print(f"📄 Processing: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        
        # Extract title and headings
        title = self._extract_title_optimized(doc)
        headings = self._extract_headings_optimized(doc)
        
        doc.close()
        
        result = {
            "title": title,
            "outline": headings
        }
        
        if self.debug:
            print(f"✅ Extracted title: '{title}'")
            print(f"✅ Found {len(headings)} headings")
        
        return result
        
    except Exception as e:
        if self.debug:
            print(f"❌ Error processing {pdf_path}: {e}")
        return {"title": "", "outline": []}

def _extract_title_optimized(self, doc: fitz.Document) -> str:
    """Optimized title extraction"""
    if not doc:
        return ""
    
    first_page = doc[0]
    page_height = first_page.rect.height
    
    # Get structured text from title area
    title_area_height = page_height * self.config['title_search_ratio']
    
    # Extract text blocks preserving structure
    blocks = first_page.get_text("dict")
    title_candidates = []
    
    for block in blocks.get("blocks", []):
        if "lines" not in block:
            continue
        
        block_bbox = block.get("bbox", [0, 0, 0, 0])
        if block_bbox[1] > title_area_height:
            continue
        
        # Process block as potential title
        block_text = self._extract_clean_block_text(block)
        
        if block_text and len(block_text) >= self.config['title_min_length']:
            # Calculate score based on position, size, and content
            score = self._calculate_title_score(block, page_height)
            title_candidates.append((score, block_text))
    
    if not title_candidates:
        return ""
    
    # Sort by score and select best
    title_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score, best_title = title_candidates[0]
    if best_score >= self.config['title_min_confidence']:
        # Clean up the title
        cleaned_title = self._clean_title(best_title)
        if not cleaned_title.endswith(' '):
            cleaned_title += ' '
        return cleaned_title
    
    return ""

def _extract_headings_optimized(self, doc: fitz.Document) -> List[Dict[str, Any]]:
    """Optimized heading extraction with text grouping"""
    if not doc:
        return []
    
    all_text_elements = []
    page_heights = {}
    
    # Collect all text elements
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_heights[page_num] = page.rect.height
        
        elements = self._extract_text_elements_optimized(page, page_num)
        all_text_elements.extend(elements)
    
    if not all_text_elements:
        return []
    
    # Group nearby elements into potential headings
    grouped_elements = self._group_nearby_elements(all_text_elements)
    
    # Analyze fonts for heading detection
    font_analysis = self._analyze_fonts_optimized(all_text_elements)
    
    # Detect headers/footers
    header_footer_texts = self._detect_repeating_elements(all_text_elements, page_heights)
    
    # Extract headings
    headings = []
    used_texts = set()
    
    for group in grouped_elements:
        combined_text = self._combine_group_text(group)
        
        if (combined_text in used_texts or 
            combined_text in header_footer_texts or
            len(combined_text) < self.config['min_heading_length'] or
            len(combined_text) > self.config['max_heading_length']):
            continue
        
        # Calculate heading confidence
        confidence = self._calculate_heading_confidence_optimized(group, font_analysis)
        
        if confidence >= self.config['heading_confidence_threshold']:
            # Determine heading level
            level = self._determine_heading_level_optimized(group[0])  # Use first element for level
            
            headings.append({
                "level": level,
                "text": combined_text,
                "page": group[0]['page']
            })
            
            used_texts.add(combined_text)
    
    # Sort by page and position
    headings.sort(key=lambda x: (x['page'], self._get_heading_y_position(x, all_text_elements)))
    
    return headings

def _extract_clean_block_text(self, block: Dict[str, Any]) -> str:
    """Extract clean text from a block, combining lines properly"""
    text_parts = []
    
    for line in block.get("lines", []):
        line_parts = []
        for span in line.get("spans", []):
            text = span.get("text", "").strip()
            if text:
                line_parts.append(text)
        
        if line_parts:
            line_text = " ".join(line_parts)
            text_parts.append(line_text)
    
    if text_parts:
        # Join lines with spaces
        combined = " ".join(text_parts)
        # Clean up multiple spaces
        combined = re.sub(r'\s+', ' ', combined).strip()
        return combined
    
    return ""

def _extract_text_elements_optimized(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
    """Extract text elements with optimization for grouping"""
    elements = []
    blocks = page.get_text("dict")
    
    for block in blocks.get("blocks", []):
        if "lines" not in block:
            continue
        
        for line in block["lines"]:
            for span in line["spans"]:
                text = span.get("text", "").strip()
                if not text or len(text) < 2:
                    continue
                
                bbox = span.get("bbox", [0, 0, 0, 0])
                
                elements.append({
                    'text': text,
                    'font': span.get('font', ''),
                    'size': span.get('size', 0),
                    'flags': span.get('flags', 0),
                    'bbox': bbox,
                    'x': bbox[0],
                    'y': bbox[1],
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1],
                    'page': page_num
                })
    
    return elements

def _group_nearby_elements(self, elements: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group nearby elements that likely form complete headings"""
    if not elements:
        return []
    
    # Sort by page, then by y, then by x
    sorted_elements = sorted(elements, key=lambda x: (x['page'], x['y'], x['x']))
    
    groups = []
    current_group = [sorted_elements[0]]
    
    for i in range(1, len(sorted_elements)):
        curr_elem = sorted_elements[i]
        prev_elem = sorted_elements[i-1]
        
        # Check if elements should be grouped
        if (curr_elem['page'] == prev_elem['page'] and
            abs(curr_elem['y'] - prev_elem['y']) <= self.config['line_grouping_threshold'] and
            curr_elem['x'] - (prev_elem['x'] + prev_elem['width']) <= self.config['text_merge_distance']):
            current_group.append(curr_elem)
        else:
            # Start new group
            if current_group:
                groups.append(current_group)
            current_group = [curr_elem]
    
    # Add last group
    if current_group:
        groups.append(current_group)
    
    return groups

def _combine_group_text(self, group: List[Dict[str, Any]]) -> str:
    """Combine text from grouped elements"""
    if not group:
        return ""
    
    # Sort by x position within the group
    sorted_group = sorted(group, key=lambda x: x['x'])
    
    text_parts = []
    for element in sorted_group:
        text = element['text'].strip()
        if text:
            text_parts.append(text)
    
    if text_parts:
        combined = " ".join(text_parts)
        # Clean up spacing
        combined = re.sub(r'\s+', ' ', combined).strip()
        
        # Add trailing space for consistency
        if not combined.endswith(' '):
            combined += ' '
        
        return combined
    
    return ""

def _calculate_title_score(self, block: Dict[str, Any], page_height: float) -> float:
    """Calculate title score for a block"""
    score = 0.0
    
    # Get block position and size info
    bbox = block.get("bbox", [0, 0, 0, 0])
    y_pos = bbox[1]
    
    # Get largest font size in block
    max_font_size = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            max_font_size = max(max_font_size, span.get('size', 0))
    
    # Position score (higher on page is better)
    position_score = 1.0 - (y_pos / (page_height * 0.4))
    score += max(position_score, 0) * 0.4
    
    # Size score
    size_score = min(max_font_size / 14.0, 2.0)
    score += size_score * 0.4
    
    # Text quality score
    block_text = self._extract_clean_block_text(block)
    quality_score = self._assess_text_quality_optimized(block_text)
    score += quality_score * 0.2
    
    return min(score, 1.0)

def _analyze_fonts_optimized(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze font patterns for heading detection"""
    font_counts = Counter()
    font_sizes = defaultdict(list)
    
    for element in elements:
        font_key = f"{element['font']}_{element['size']:.1f}"
        font_counts[font_key] += 1
        font_sizes[font_key].append(element['size'])
    
    total_elements = len(elements)
    
    # Calculate font statistics
    font_stats = {}
    for font_key, count in font_counts.items():
        frequency = count / total_elements
        avg_size = sum(font_sizes[font_key]) / len(font_sizes[font_key])
        
        font_stats[font_key] = {
            'frequency': frequency,
            'count': count,
            'avg_size': avg_size
        }
    
    return font_stats

def _detect_repeating_elements(self, elements: List[Dict[str, Any]], page_heights: Dict[int, float]) -> Set[str]:
    """Detect headers/footers that repeat across pages"""
    text_pages = defaultdict(set)
    
    for element in elements:
        text = element['text'].strip()
        if len(text) >= 3:
            text_pages[text].add(element['page'])
    
    # Find texts that appear on multiple pages
    repeating_texts = set()
    for text, pages in text_pages.items():
        if len(pages) > 1:
            repeating_texts.add(text)
    
    return repeating_texts

def _calculate_heading_confidence_optimized(self, group: List[Dict[str, Any]], font_analysis: Dict[str, Any]) -> float:
    """Calculate optimized heading confidence for a group"""
    if not group:
        return 0.0
    
    score = 0.0
    
    # Use the largest element in the group for analysis
    main_element = max(group, key=lambda x: x['size'])
    font_key = f"{main_element['font']}_{main_element['size']:.1f}"
    
    # Font rarity score
    if font_key in font_analysis:
        frequency = font_analysis[font_key]['frequency']
        rarity_score = max(0, 1.0 - frequency * 2.5)  # Penalize high frequency
        score += rarity_score * 0.4
    
    # Size score
    font_size = main_element['size']
    if font_size >= self.config['min_font_size']:
        size_score = min((font_size - 8) / 8, 1.0)  # Normalize from size 8
        score += size_score * 0.3
    
    # Text quality score
    combined_text = self._combine_group_text(group)
    quality_score = self._assess_text_quality_optimized(combined_text)
    score += quality_score * 0.3
    
    return min(score, 1.0)

def _assess_text_quality_optimized(self, text: str) -> float:
    """Assess text quality for titles and headings"""
    if not text:
        return 0.0
    
    score = 0.0
    
    # Length factor
    length_factor = min(len(text) / 30.0, 1.0)
    score += length_factor * 0.3
    
    # Word structure
    words = text.split()
    if words:
        # Prefer moderate number of words
        word_count_factor = min(len(words) / 6.0, 1.0)
        score += word_count_factor * 0.3
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        word_length_factor = min(avg_word_length / 5.0, 1.0)
        score += word_length_factor * 0.2
    
    # Capitalization
    if text and text[0].isupper():
        score += 0.2
    
    return min(score, 1.0)

def _determine_heading_level_optimized(self, element: Dict[str, Any]) -> str:
    """Determine heading level based on font size"""
    font_size = element['size']
    
    if font_size >= 16:
        return "H1"
    elif font_size >= 13:
        return "H2"
    elif font_size >= 11:
        return "H3"
    else:
        return "H4"

def _clean_title(self, title: str) -> str:
    """Clean up title text"""
    if not title:
        return ""
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', title.strip())
    
    # Remove repeated segments (for file03 issue)
    words = cleaned.split()
    if len(words) > 6:
        # Check for repetitive patterns
        unique_words = []
        seen = set()
        for word in words:
            if word.lower() not in seen or len(unique_words) < 3:
                unique_words.append(word)
                seen.add(word.lower())
        
        if len(unique_words) < len(words) * 0.7:  # If too much repetition
            cleaned = " ".join(unique_words[:10])  # Take first 10 unique words
    
    return cleaned

def _get_heading_y_position(self, heading: Dict[str, Any], all_elements: List[Dict[str, Any]]) -> float:
    """Get y position for heading sorting"""
    # Find the corresponding element to get actual y position
    for element in all_elements:
        if (element['page'] == heading['page'] and 
            heading['text'].strip().startswith(element['text'].strip())):
            return element['y']
    return 0.0
