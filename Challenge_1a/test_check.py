from check import process_pdf_for_hackathon
import json

# Test the function with a sample PDF
try:
    result = process_pdf_for_hackathon('sample_dataset/pdfs/file01.pdf')
    print("SUCCESS: PDF processing completed")
    print("\nResult:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"ERROR: {e}")
