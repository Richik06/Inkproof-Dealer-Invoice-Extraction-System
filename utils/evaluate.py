#!/usr/bin/env python3
"""
Evaluation script for measuring accuracy against ground truth
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
from utils.validation import (
    fuzzy_match_score,
    calculate_iou,
    normalize_text
)


def load_ground_truth(gt_file: str) -> Dict:
    """Load ground truth annotations"""
    with open(gt_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_predictions(pred_file: str) -> Dict:
    """Load prediction results"""
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert list to dict if needed
    if isinstance(data, list):
        return {item['doc_id']: item for item in data}
    return data


def evaluate_field(pred_value, gt_value, field_type: str, tolerance: float = 0.05) -> Dict:
    """
    Evaluate a single field
    Returns dict with is_correct and score
    """
    result = {"is_correct": False, "score": 0.0, "error": None}
    
    if field_type == "dealer_name":
        # Fuzzy match with 90% threshold
        if pred_value and gt_value:
            score = fuzzy_match_score(pred_value, gt_value)
            result["score"] = score
            result["is_correct"] = score >= 0.90
        else:
            result["error"] = "Missing value"
    
    elif field_type == "model_name":
        # Exact match (case-insensitive)
        if pred_value and gt_value:
            pred_norm = normalize_text(pred_value)
            gt_norm = normalize_text(gt_value)
            result["is_correct"] = pred_norm == gt_norm
            result["score"] = 1.0 if result["is_correct"] else 0.0
        else:
            result["error"] = "Missing value"
    
    elif field_type in ["horse_power", "asset_cost"]:
        # Numeric with tolerance
        if pred_value is not None and gt_value is not None:
            diff_ratio = abs(pred_value - gt_value) / max(gt_value, 1)
            result["is_correct"] = diff_ratio <= tolerance
            result["score"] = 1.0 - min(diff_ratio, 1.0)
        else:
            result["error"] = "Missing value"
    
    elif field_type in ["signature", "stamp"]:
        # Presence + IoU for bounding box
        pred_present = pred_value.get("present", False)
        gt_present = gt_value.get("present", False)
        
        # Check presence
        presence_correct = pred_present == gt_present
        
        # Check IoU if both present
        iou = 0.0
        if pred_present and gt_present:
            pred_bbox = pred_value.get("bbox", [0, 0, 0, 0])
            gt_bbox = gt_value.get("bbox", [0, 0, 0, 0])
            iou = calculate_iou(pred_bbox, gt_bbox)
        
        result["is_correct"] = presence_correct and (not gt_present or iou >= 0.5)
        result["score"] = iou if gt_present else (1.0 if presence_correct else 0.0)
        result["iou"] = iou
    
    return result


def evaluate_document(prediction: Dict, ground_truth: Dict) -> Dict:
    """
    Evaluate all fields for a single document
    Returns evaluation metrics
    """
    doc_id = prediction.get("doc_id", "unknown")
    pred_fields = prediction.get("fields", {})
    gt_fields = ground_truth.get("fields", {})
    
    field_results = {}
    
    # Evaluate each field
    field_results["dealer_name"] = evaluate_field(
        pred_fields.get("dealer_name"),
        gt_fields.get("dealer_name"),
        "dealer_name"
    )
    
    field_results["model_name"] = evaluate_field(
        pred_fields.get("model_name"),
        gt_fields.get("model_name"),
        "model_name"
    )
    
    field_results["horse_power"] = evaluate_field(
        pred_fields.get("horse_power"),
        gt_fields.get("horse_power"),
        "horse_power"
    )
    
    field_results["asset_cost"] = evaluate_field(
        pred_fields.get("asset_cost"),
        gt_fields.get("asset_cost"),
        "asset_cost"
    )
    
    field_results["signature"] = evaluate_field(
        pred_fields.get("signature"),
        gt_fields.get("signature"),
        "signature"
    )
    
    field_results["stamp"] = evaluate_field(
        pred_fields.get("stamp"),
        gt_fields.get("stamp"),
        "stamp"
    )
    
    # Calculate document-level accuracy
    all_correct = all(r["is_correct"] for r in field_results.values())
    
    return {
        "doc_id": doc_id,
        "document_level_correct": all_correct,
        "field_results": field_results,
        "confidence": prediction.get("confidence", 0.0),
        "processing_time": prediction.get("processing_time_sec", 0.0)
    }


def evaluate_batch(predictions: Dict, ground_truth: Dict) -> Dict:
    """
    Evaluate all documents and compute aggregate metrics
    """
    results = []
    
    for doc_id, gt in ground_truth.items():
        if doc_id not in predictions:
            print(f"Warning: No prediction for {doc_id}")
            continue
        
        pred = predictions[doc_id]
        doc_result = evaluate_document(pred, gt)
        results.append(doc_result)
    
    # Calculate aggregate metrics
    total_docs = len(results)
    
    if total_docs == 0:
        return {"error": "No documents to evaluate"}
    
    # Document-Level Accuracy (DLA)
    dla = sum(r["document_level_correct"] for r in results) / total_docs
    
    # Field-level accuracies
    field_accuracies = {}
    for field in ["dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"]:
        correct = sum(r["field_results"][field]["is_correct"] for r in results)
        field_accuracies[field] = correct / total_docs
    
    # Average confidence
    avg_confidence = sum(r["confidence"] for r in results) / total_docs
    
    # Average processing time
    avg_time = sum(r["processing_time"] for r in results) / total_docs
    
    # Confidence vs Accuracy correlation
    high_conf_results = [r for r in results if r["confidence"] > 0.9]
    high_conf_dla = (sum(r["document_level_correct"] for r in high_conf_results) / 
                     len(high_conf_results) if high_conf_results else 0)
    
    metrics = {
        "total_documents": total_docs,
        "document_level_accuracy": dla,
        "field_accuracies": field_accuracies,
        "average_confidence": avg_confidence,
        "average_processing_time": avg_time,
        "high_confidence_accuracy": high_conf_dla,
        "high_confidence_count": len(high_conf_results),
        "detailed_results": results
    }
    
    return metrics


def print_evaluation_report(metrics: Dict):
    """Print detailed evaluation report"""
    print(f"\n{'='*70}")
    print("EVALUATION REPORT")
    print(f"{'='*70}")
    
    print(f"\nOverall Metrics:")
    print(f"  Total Documents: {metrics['total_documents']}")
    print(f"  Document-Level Accuracy (DLA): {metrics['document_level_accuracy']:.1%}")
    print(f"  Average Confidence: {metrics['average_confidence']:.3f}")
    print(f"  Average Processing Time: {metrics['average_processing_time']:.2f}s")
    
    print(f"\nField-Level Accuracies:")
    for field, acc in metrics['field_accuracies'].items():
        print(f"  {field:15s}: {acc:.1%}")
    
    print(f"\nHigh Confidence Performance (conf > 0.9):")
    print(f"  Count: {metrics['high_confidence_count']}/{metrics['total_documents']}")
    print(f"  Accuracy: {metrics['high_confidence_accuracy']:.1%}")
    
    # Performance breakdown
    print(f"\n{'='*70}")
    print("Performance Breakdown:")
    print(f"{'='*70}")
    
    dla = metrics['document_level_accuracy']
    if dla >= 0.95:
        print("✓ EXCELLENT: DLA ≥ 95% (Target Met)")
    elif dla >= 0.90:
        print("✓ GOOD: DLA ≥ 90% (Close to Target)")
    elif dla >= 0.80:
        print("⚠ FAIR: DLA ≥ 80% (Needs Improvement)")
    else:
        print("✗ POOR: DLA < 80% (Significant Improvement Needed)")


def save_evaluation_results(metrics: Dict, output_file: str):
    """Save evaluation results to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Also save CSV for detailed analysis
    if "detailed_results" in metrics:
        df_data = []
        for result in metrics["detailed_results"]:
            row = {
                "doc_id": result["doc_id"],
                "document_correct": result["document_level_correct"],
                "confidence": result["confidence"],
                "processing_time": result["processing_time"]
            }
            # Add field results
            for field, field_result in result["field_results"].items():
                row[f"{field}_correct"] = field_result["is_correct"]
                row[f"{field}_score"] = field_result["score"]
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nDetailed results saved to: {csv_file}")


def main():
    """Main evaluation function"""
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <predictions.json> <ground_truth.json> [output.json]")
        sys.exit(1)
    
    pred_file = sys.argv[1]
    gt_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "evaluation_results.json"
    
    print("Loading data...")
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(gt_file)
    
    print(f"Evaluating {len(ground_truth)} documents...")
    metrics = evaluate_batch(predictions, ground_truth)
    
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        sys.exit(1)
    
    print_evaluation_report(metrics)
    
    save_evaluation_results(metrics, output_file)
    print(f"\nEvaluation results saved to: {output_file}")


if __name__ == "__main__":
    main()