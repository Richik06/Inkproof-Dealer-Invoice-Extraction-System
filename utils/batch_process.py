#!/usr/bin/env python3
"""
Batch processing script for multiple documents
Includes progress tracking and error handling
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from executable import DocumentExtractor


def process_batch(input_dir: str, output_dir: str = "output", model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
    """
    Process all PDFs in a directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Initialize extractor
    print("Loading model...")
    extractor = DocumentExtractor(model_name=model_name)
    
    # Process files
    results = []
    successful = 0
    failed = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing documents"):
        try:
            result = extractor.process_document(str(pdf_file))
            results.append(result)
            
            if result["confidence"] > 0.5:
                successful += 1
            else:
                failed += 1
                
            # Save individual result
            result_file = output_path / f"{pdf_file.stem}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"\nError processing {pdf_file.name}: {e}")
            failed += 1
            results.append({
                "doc_id": pdf_file.stem,
                "error": str(e),
                "confidence": 0.0
            })
    
    # Save combined results
    combined_file = output_path / "all_results.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate summary report
    generate_summary_report(results, output_path)
    
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Total documents: {len(pdf_files)}")
    print(f"Successful (conf > 0.5): {successful}")
    print(f"Failed or low confidence: {failed}")
    print(f"Results saved to: {output_path}")


def generate_summary_report(results: list, output_path: Path):
    """Generate summary statistics and reports"""
    
    # Extract metrics
    metrics = []
    for result in results:
        if "error" not in result:
            metrics.append({
                "doc_id": result["doc_id"],
                "confidence": result["confidence"],
                "processing_time": result["processing_time_sec"],
                "dealer_name": result["fields"]["dealer_name"],
                "model_name": result["fields"]["model_name"],
                "horse_power": result["fields"]["horse_power"],
                "asset_cost": result["fields"]["asset_cost"],
                "signature_present": result["fields"]["signature"]["present"],
                "stamp_present": result["fields"]["stamp"]["present"]
            })
    
    if not metrics:
        print("No successful extractions to summarize")
        return
    
    df = pd.DataFrame(metrics)
    
    # Calculate statistics
    stats = {
        "total_documents": len(results),
        "successful_extractions": len(metrics),
        "average_confidence": float(df["confidence"].mean()),
        "median_confidence": float(df["confidence"].median()),
        "min_confidence": float(df["confidence"].min()),
        "max_confidence": float(df["confidence"].max()),
        "average_processing_time": float(df["processing_time"].mean()),
        "median_processing_time": float(df["processing_time"].median()),
        "field_completeness": {
            "dealer_name": int((df["dealer_name"] != "").sum()),
            "model_name": int((df["model_name"] != "").sum()),
            "horse_power": int(df["horse_power"].notna().sum()),
            "asset_cost": int(df["asset_cost"].notna().sum()),
            "signature": int(df["signature_present"].sum()),
            "stamp": int(df["stamp_present"].sum())
        },
        "confidence_distribution": {
            "excellent (>0.9)": int((df["confidence"] > 0.9).sum()),
            "good (0.7-0.9)": int(((df["confidence"] >= 0.7) & (df["confidence"] <= 0.9)).sum()),
            "fair (0.5-0.7)": int(((df["confidence"] >= 0.5) & (df["confidence"] < 0.7)).sum()),
            "poor (<0.5)": int((df["confidence"] < 0.5).sum())
        }
    }
    
    # Save statistics
    stats_file = output_path / "summary_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Save CSV for analysis
    csv_file = output_path / "extraction_results.csv"
    df.to_csv(csv_file, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Successful: {stats['successful_extractions']}")
    print(f"\nConfidence Metrics:")
    print(f"  Average: {stats['average_confidence']:.3f}")
    print(f"  Median: {stats['median_confidence']:.3f}")
    print(f"  Range: [{stats['min_confidence']:.3f}, {stats['max_confidence']:.3f}]")
    print(f"\nProcessing Time:")
    print(f"  Average: {stats['average_processing_time']:.2f}s")
    print(f"  Median: {stats['median_processing_time']:.2f}s")
    print(f"\nConfidence Distribution:")
    for category, count in stats['confidence_distribution'].items():
        print(f"  {category}: {count}")
    print(f"\nField Completeness:")
    for field, count in stats['field_completeness'].items():
        percentage = (count / stats['successful_extractions']) * 100
        print(f"  {field}: {count}/{stats['successful_extractions']} ({percentage:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_process.py <input_directory> [output_directory] [model_name]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2-VL-2B-Instruct"
    
    process_batch(input_dir, output_dir, model_name)