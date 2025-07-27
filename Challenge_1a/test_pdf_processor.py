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

# Test the function
if __name__ == "__main__":
    try:
        result = process_pdf_for_hackathon('sample_dataset/pdfs/file01.pdf')
        print("SUCCESS: PDF processing completed")
        print("\nResult:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
