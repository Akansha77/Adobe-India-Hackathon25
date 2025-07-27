from check import process_pdf_for_hackathon
import json

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
    print(f"  Match: {'✅ YES' if title_match else '❌ NO'}")
    
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
    if exact_accuracy < 100:
        print(f"\nFIRST FEW EXPECTED VS ACTUAL:")
        for i, expected_item in enumerate(expected_outline[:5]):
            print(f"  Expected[{i}]: {expected_item['level']} - '{expected_item['text'][:50]}...' (P{expected_item['page']})")
        print(f"  ---")
        for i, check_item in enumerate(check_outline[:5]):
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
        print(f"  {r['filename']}: Title {'✅' if r['title_match'] else '❌'} | Outline {r['exact_accuracy']:.1f}% exact | {r['expected_count']} expected vs {r['actual_count']} actual")
