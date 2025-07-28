#!/usr/bin/env python3
"""
Simple runner script for Challenge 1B collections - Clean version
"""

import subprocess
import sys
import os
from pathlib import Path

def run_collection(collection_name: str):
    """Run processing for a specific collection"""
    print(f"\n=> Processing {collection_name}...")
    
    collection_path = Path(collection_name)
    if not collection_path.exists():
        print(f"ERROR: Collection path does not exist: {collection_path}")
        return False
    
    # Determine the processor script name
    collection_num = collection_name.split()[-1]  # Get "1", "2", or "3"
    processor_script = f"collection{collection_num}_processor.py"
    processor_path = collection_path / processor_script
    
    if not processor_path.exists():
        print(f"ERROR: Processor script not found: {processor_path}")
        return False
    
    try:
        # Set environment to avoid encoding issues
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run the specific collection processor from within the collection directory
        result = subprocess.run([
            sys.executable, processor_script
        ], cwd=collection_path, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print(f"SUCCESS: {collection_name} processed successfully")
            
            # Check if output file was created
            output_file = collection_path / f"collection{collection_num}_output.json"
            if output_file.exists():
                print(f"   OUTPUT: {output_file}")
                return True
            else:
                print(f"   WARNING: Output file not found: {output_file}")
                return False
        else:
            print(f"ERROR: Failed to process {collection_name}")
            return False
            
    except Exception as e:
        print(f"ERROR: Exception processing {collection_name}: {e}")
        return False

def main():
    """Run all collections"""
    print("Adobe Hackathon 2025 - Challenge 1B Multi-Collection Processing")
    print("=" * 70)
    
    collections = ["Collection 1", "Collection 2", "Collection 3"]
    success_count = 0
    
    for collection in collections:
        if run_collection(collection):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"PROCESSING SUMMARY:")
    print(f"   Successful: {success_count}/{len(collections)} collections")
    
    if success_count == len(collections):
        print("   All collections processed successfully!")
    else:
        print(f"   {len(collections) - success_count} collection(s) failed")
    
    return 0 if success_count == len(collections) else 1

if __name__ == "__main__":
    exit(main())
