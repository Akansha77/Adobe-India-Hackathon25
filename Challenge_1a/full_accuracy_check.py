import fitz  # PyMuPDF
import re
import json

def clean_text(text):
    return text.strip().replace('\n', ' ').replace('\xa0', ' ')

def cluster_font_sizes(font_sizes):
    unique_sizes = sorted(set(font_sizes), reverse=True)
    level_map = {}
    if unique_sizes:
        level_map[unique_sizes[0]] = "H1"
    if len(unique_sizes) > 1:
        level_map[unique_sizes[1]] = "H2"
    if len(unique_sizes) > 2:
        level_map[unique_sizes[2]] = "H3"
    if len(unique_sizes) > 3:
        level_map[unique_sizes[3]] = "H4"
    return level_map

def is_likely_heading(text):
    text = text.strip()
    if not text or len(text) > 200:
        return False
    if re.match(r"^\d{1,2}\s+[A-Z]{3,}\s+\d{4}$", text):  # Date like "18 JUNE 2024"
        return False
    words = text.split()
    cap_words = sum(1 for w in words if w.istitle() or w.isupper())
    return cap_words / len(words) >= 0.3 or text.endswith(":")

def classify_heading_by_number(text):
    if re.match(r"^\d+\.\d+\.\d+", text):
        return "H4"
    elif re.match(r"^\d+\.\d+", text):
        return "H3"
    elif re.match(r"^\d+", text):
        return "H2"
    return None

def process_pdf_for_hackathon(file_path):
    doc = fitz.open(file_path)
    all_spans = []

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                line_text = ""
                font_size_sum = 0
                span_count = 0
                for span in line.get("spans", []):
                    text = clean_text(span["text"])
                    if text:
                        line_text += " " + text
                        font_size_sum += span["size"]
                        span_count += 1
                if line_text.strip() and span_count > 0:
                    avg_size = font_size_sum / span_count
                    all_spans.append({
                        "text": line_text.strip(),
                        "size": round(avg_size, 1),
                        "page": page_num,
                        "y": line["bbox"][1]
                    })

    font_sizes = [s["size"] for s in all_spans]
    level_map = cluster_font_sizes(font_sizes)

    # Fix title: merge top stacked lines of largest font size on first page
    first_page = [s for s in all_spans if s["page"] == 1 and s["size"] == max(font_sizes)]
    sorted_first_page = sorted(first_page, key=lambda x: x["y"])
    title_parts = []
    for s in sorted_first_page:
        if s["text"] not in title_parts:
            title_parts.append(s["text"])
    title = " ".join(title_parts)

    # Build outline
    outline = []
    prev_y_by_page = {}
    for item in all_spans:
        text = item["text"]
        page = item["page"]
        size = item["size"]
        y = item["y"]

        heading_level = classify_heading_by_number(text)
        if not heading_level:
            heading_level = level_map.get(size)

        if heading_level and is_likely_heading(text):
            # De-duplicate lines close in y-pos (visual proximity)
            last_y = prev_y_by_page.get((page, heading_level))
            if last_y and abs(y - last_y) < 8:
                continue
            prev_y_by_page[(page, heading_level)] = y
            outline.append({
                "level": heading_level,
                "text": text.strip() + " ",
                "page": page
            })

    return {
        "title": title.strip() + "  ",
        "outline": outline
    }

def compare_outputs(filename):
    # Get check.py output
    try:
        check_result = process_pdf_for_hackathon(f'sample_dataset/pdfs/{filename}')
    except Exception as e:
        print(f"ERROR in check.py for {filename}: {e}")
        return
    
    # Get expected output
    try:
        with open(f'sample_dataset/outputs/{filename.replace(".pdf", ".json")}', 'r', encoding='utf-8') as f:
            expected_result = json.load(f)
    except Exception as e:
        print(f"ERROR reading expected output for {filename}: {e}")
        return
    
    print(f"\n{'='*60}")
    print(f"ACCURACY CHECK FOR {filename}")
    print('='*60)
    
    # Compare titles
    check_title = check_result['title'].strip()
    expected_title = expected_result['title'].strip()
    title_match = check_title == expected_title
    
    print(f"TITLE COMPARISON:")
    print(f"  Expected: '{expected_title}'")
    print(f"  Check.py: '{check_title}'")
    print(f"  Match: {'SUCCESS' if title_match else 'FAIL'}")
    
    # Compare outlines
    check_outline = check_result['outline']
    expected_outline = expected_result['outline']
    
    print(f"\nOUTLINE COMPARISON:")
    print(f"  Expected items: {len(expected_outline)}")
    print(f"  Check.py items: {len(check_outline)}")
    
    # Find matches
    exact_matches = 0
    partial_matches = 0
    
    for expected_item in expected_outline:
        found_exact = False
        found_partial = False
        
        for check_item in check_outline:
            # Exact match (level, text, page)
            if (expected_item['level'] == check_item['level'] and 
                expected_item['text'].strip() == check_item['text'].strip() and 
                expected_item['page'] == check_item['page']):
                exact_matches += 1
                found_exact = True
                break
            # Partial match (similar text, same page)
            elif (expected_item['page'] == check_item['page'] and 
                  expected_item['text'].strip().lower() in check_item['text'].strip().lower() or
                  check_item['text'].strip().lower() in expected_item['text'].strip().lower()):
                if not found_partial:
                    partial_matches += 1
                    found_partial = True
    
    # Calculate accuracy
    if len(expected_outline) > 0:
        exact_accuracy = (exact_matches / len(expected_outline)) * 100
        partial_accuracy = ((exact_matches + partial_matches) / len(expected_outline)) * 100
    else:
        exact_accuracy = 100 if len(check_outline) == 0 else 0
        partial_accuracy = exact_accuracy
    
    print(f"  Exact matches: {exact_matches}/{len(expected_outline)} ({exact_accuracy:.1f}%)")
    print(f"  Partial matches: {partial_matches}")
    print(f"  Overall accuracy: {partial_accuracy:.1f}%")
    
    # Show first few mismatches for debugging
    if exact_accuracy < 100 and len(expected_outline) > 0:
        print(f"\nFIRST FEW EXPECTED VS ACTUAL:")
        for i, expected_item in enumerate(expected_outline[:3]):
            print(f"  Expected[{i}]: {expected_item['level']} - '{expected_item['text'][:50]}...' (P{expected_item['page']})")
        print(f"  ---")
        for i, check_item in enumerate(check_outline[:3]):
            print(f"  Check.py[{i}]: {check_item['level']} - '{check_item['text'][:50]}...' (P{check_item['page']})")
    
    return {
        'filename': filename,
        'title_match': title_match,
        'exact_accuracy': exact_accuracy,
        'partial_accuracy': partial_accuracy,
        'expected_count': len(expected_outline),
        'actual_count': len(check_outline)
    }

# Test all files
test_files = ['file01.pdf', 'file02.pdf', 'file03.pdf', 'file04.pdf', 'file05.pdf']
results = []

for filename in test_files:
    result = compare_outputs(filename)
    if result:
        results.append(result)

# Overall summary
print(f"\n{'='*60}")
print("OVERALL ACCURACY SUMMARY")
print('='*60)

if results:
    avg_title_accuracy = sum(1 for r in results if r['title_match']) / len(results) * 100
    avg_exact_accuracy = sum(r['exact_accuracy'] for r in results) / len(results)
    avg_partial_accuracy = sum(r['partial_accuracy'] for r in results) / len(results)
    
    print(f"Files tested: {len(results)}")
    print(f"Title accuracy: {avg_title_accuracy:.1f}%")
    print(f"Outline exact accuracy: {avg_exact_accuracy:.1f}%")
    print(f"Outline partial accuracy: {avg_partial_accuracy:.1f}%")
    
    print(f"\nPER-FILE BREAKDOWN:")
    for r in results:
        status = 'SUCCESS' if r['title_match'] else 'FAIL'
        print(f"  {r['filename']}: Title {status} | Outline {r['exact_accuracy']:.1f}% exact | {r['expected_count']} expected vs {r['actual_count']} actual")
