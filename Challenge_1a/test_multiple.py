exec(open('test_pdf_processor.py').read())

# Test on multiple files
test_files = ['file01.pdf', 'file02.pdf', 'file03.pdf']

for filename in test_files:
    filepath = f'sample_dataset/pdfs/{filename}'
    try:
        print(f"\n{'='*50}")
        print(f"PROCESSING: {filename}")
        print('='*50)
        result = process_pdf_for_hackathon(filepath)
        print(f"Title: {result['title']}")
        print(f"Outline items: {len(result['outline'])}")
        print("\nOutline preview (first 5 items):")
        for i, item in enumerate(result['outline'][:5]):
            print(f"  {item['level']}: {item['text'][:60]}... (Page {item['page']})")
        if len(result['outline']) > 5:
            print(f"  ... and {len(result['outline']) - 5} more items")
    except Exception as e:
        print(f"ERROR processing {filename}: {e}")
