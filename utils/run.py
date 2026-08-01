"""
Simple runner script - Just edit the paths below and run!
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from executable import HybridExtractor
from pathlib import Path
import json
import time



# Input: Your PDF/PNG files location
# Examples:
# INPUT_PATH = r"C:\Users\vichi\document-ai-extractor\data"  # Directory
# INPUT_PATH = r"C:\Users\vichi\document-ai-extractor\data\invoice.pdf"  # Single file
INPUT_PATH = r"data"  # Relative path to data folder

# Output: Where to save results (must end with .json)
OUTPUT_PATH = r"output\results.json"

# Model: Choose one
# Fast (2B): "Qwen/Qwen2.5-VL-2B-Instruct"
# Accurate (7B): "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # Use 7B for better accuracy


def main():
    print("\n" + "="*70)
    print("DOCUMENT AI FIELD EXTRACTION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Input:  {INPUT_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Model:  {MODEL_NAME}")
    print("="*70 + "\n")
    
    try:
        # Initialize
        print("Initializing extractor...")
        extractor = HybridExtractor(model_name=MODEL_NAME)
        
        # Process
        input_path = Path(INPUT_PATH)
        results = []
        
        if not input_path.exists():
            print(f"ERROR: Path does not exist: {INPUT_PATH}")
            print("Please check the INPUT_PATH in run.py")
            return
        
        if input_path.is_file():
            # Single file
            print(f"\nProcessing single file...")
            result = extractor.process_document(str(input_path))
            results.append(result)
            
        elif input_path.is_dir():
            # Directory
            files = (list(input_path.glob('*.pdf')) + 
                    list(input_path.glob('*.png')) + 
                    list(input_path.glob('*.jpg')) +
                    list(input_path.glob('*.jpeg')))
            
            if not files:
                print(f"\nNo PDF or image files found in: {input_path}")
                print("Please add some files to the data/ folder")
                return
            
            print(f"\nFound {len(files)} file(s)\n")
            
            for i, file_path in enumerate(files, 1):
                print(f"\n{'='*70}")
                print(f"File {i}/{len(files)}")
                result = extractor.process(str(file_path))
                results.append(result)
        
        # Save results
        output_path = Path(OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"✓ RESULTS SAVED TO: {OUTPUT_PATH}")
        print(f"{'='*70}")
        
        # Summary
        if results:
            successful = sum(1 for r in results if r['confidence'] > 0.5)
            avg_conf = sum(r['confidence'] for r in results) / len(results)
            avg_time = sum(r['processing_time_sec'] for r in results) / len(results)
            
            print(f"\nSUMMARY:")
            print(f"  Total Documents:     {len(results)}")
            print(f"  Successful:          {successful}")
            print(f"  Failed/Low Conf:     {len(results) - successful}")
            print(f"  Average Confidence:  {avg_conf:.3f}")
            print(f"  Average Time:        {avg_time:.2f}s")
            print(f"{'='*70}\n")
            
            # Show first result sample
            if results and results[0]['confidence'] > 0:
                print("\nSAMPLE RESULT (First Document):")
                print(f"  Dealer:  {results[0]['fields']['dealer_name']}")
                print(f"  Model:   {results[0]['fields']['model_name']}")
                print(f"  HP:      {results[0]['fields']['horse_power']}")
                print(f"  Cost:    {results[0]['fields']['asset_cost']}")
                print(f"  Conf:    {results[0]['confidence']:.2f}")
                print()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {e}")
        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()