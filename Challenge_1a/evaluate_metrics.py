#!/usr/bin/env python3
"""
Adobe Hackathon 2025 - Precision, Accuracy, and Recall Evaluation
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

def load_json_file(file_path: Path) -> Dict:
    """Load JSON file safely"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return {}

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    return text.strip().lower()

def evaluate_titles(predicted_dir: Path, ground_truth_dir: Path) -> Tuple[int, int]:
    """Evaluate title extraction accuracy"""
    total_files = 0
    correct_titles = 0
    
    print("🏷️  TITLE EVALUATION:")
    print("-" * 40)
    
    for json_file in sorted(predicted_dir.glob("*.json")):
        filename = json_file.name
        gt_file = ground_truth_dir / filename
        
        if not gt_file.exists():
            continue
            
        total_files += 1
        
        predicted = load_json_file(json_file)
        ground_truth = load_json_file(gt_file)
        
        pred_title = normalize_text(predicted.get('title', ''))
        gt_title = normalize_text(ground_truth.get('title', ''))
        
        is_correct = pred_title == gt_title
        if is_correct:
            correct_titles += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{filename:12} | {status} | Match: {is_correct}")
        
        if not is_correct:
            print(f"   Predicted: '{predicted.get('title', '')[:60]}'")
            print(f"   Expected:  '{ground_truth.get('title', '')[:60]}'")
    
    return correct_titles, total_files

def evaluate_headings(predicted_dir: Path, ground_truth_dir: Path) -> Dict:
    """Evaluate heading extraction with precision, recall, and F1-score"""
    total_predicted = 0
    total_ground_truth = 0
    total_correct = 0
    
    file_results = []
    
    print("\n📋 HEADING EVALUATION:")
    print("-" * 40)
    
    for json_file in sorted(predicted_dir.glob("*.json")):
        filename = json_file.name
        gt_file = ground_truth_dir / filename
        
        if not gt_file.exists():
            continue
        
        predicted = load_json_file(json_file)
        ground_truth = load_json_file(gt_file)
        
        pred_headings = predicted.get('outline', [])
        gt_headings = ground_truth.get('outline', [])
        
        # Count total headings
        num_predicted = len(pred_headings)
        num_ground_truth = len(gt_headings)
        
        total_predicted += num_predicted
        total_ground_truth += num_ground_truth
        
        # Find correct headings (match by text and page)
        correct_headings = 0
        matched_gt = set()
        
        for pred_h in pred_headings:
            pred_text = normalize_text(pred_h.get('text', ''))
            pred_page = pred_h.get('page', -1)
            
            for i, gt_h in enumerate(gt_headings):
                if i in matched_gt:
                    continue
                    
                gt_text = normalize_text(gt_h.get('text', ''))
                gt_page = gt_h.get('page', -1)
                
                # Exact match on text and page
                if pred_text == gt_text and pred_page == gt_page:
                    correct_headings += 1
                    matched_gt.add(i)
                    break
        
        total_correct += correct_headings
        
        # Calculate per-file metrics
        precision = correct_headings / num_predicted if num_predicted > 0 else 0
        recall = correct_headings / num_ground_truth if num_ground_truth > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        file_results.append({
            'filename': filename,
            'predicted': num_predicted,
            'ground_truth': num_ground_truth,
            'correct': correct_headings,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"{filename:12} | P: {num_predicted:2d} | GT: {num_ground_truth:2d} | C: {correct_headings:2d} | Prec: {precision:.1%} | Rec: {recall:.1%} | F1: {f1:.1%}")
    
    # Calculate overall metrics
    overall_precision = total_correct / total_predicted if total_predicted > 0 else 0
    overall_recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    return {
        'total_predicted': total_predicted,
        'total_ground_truth': total_ground_truth,
        'total_correct': total_correct,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'file_results': file_results
    }

def main():
    """Main evaluation function"""
    print("🧪 ADOBE HACKATHON 2025 - PERFORMANCE EVALUATION")
    print("=" * 60)
    
    predicted_dir = Path("model_outputs")
    ground_truth_dir = Path("sample_dataset/outputs")
    
    if not predicted_dir.exists():
        print("❌ Model outputs directory not found")
        return
    
    if not ground_truth_dir.exists():
        print("❌ Ground truth directory not found")
        return
    
    # Evaluate titles
    correct_titles, total_files = evaluate_titles(predicted_dir, ground_truth_dir)
    title_accuracy = correct_titles / total_files if total_files > 0 else 0
    
    # Evaluate headings
    heading_metrics = evaluate_headings(predicted_dir, ground_truth_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 OVERALL PERFORMANCE SUMMARY:")
    print("=" * 60)
    
    print(f"🏷️  TITLE METRICS:")
    print(f"   ✓ Accuracy: {title_accuracy:.1%} ({correct_titles}/{total_files})")
    
    print(f"\n📋 HEADING METRICS:")
    print(f"   ✓ Precision: {heading_metrics['precision']:.1%}")
    print(f"   ✓ Recall: {heading_metrics['recall']:.1%}")
    print(f"   ✓ F1-Score: {heading_metrics['f1_score']:.1%}")
    
    print(f"\n📁 DETAILED COUNTS:")
    print(f"   • Files Processed: {total_files}")
    print(f"   • Total Predicted Headings: {heading_metrics['total_predicted']}")
    print(f"   • Total Ground Truth Headings: {heading_metrics['total_ground_truth']}")
    print(f"   • Correctly Matched Headings: {heading_metrics['total_correct']}")
    
    # Performance assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    if title_accuracy >= 0.9:
        print(f"   🏆 EXCELLENT: Title accuracy ≥90%")
    elif title_accuracy >= 0.8:
        print(f"   ✅ GOOD: Title accuracy ≥80%")
    elif title_accuracy >= 0.7:
        print(f"   ⚠️  FAIR: Title accuracy ≥70%")
    else:
        print(f"   ❌ POOR: Title accuracy <70%")
    
    if heading_metrics['f1_score'] >= 0.8:
        print(f"   🏆 EXCELLENT: Heading F1-score ≥80%")
    elif heading_metrics['f1_score'] >= 0.7:
        print(f"   ✅ GOOD: Heading F1-score ≥70%")
    elif heading_metrics['f1_score'] >= 0.6:
        print(f"   ⚠️  FAIR: Heading F1-score ≥60%")
    else:
        print(f"   ❌ POOR: Heading F1-score <60%")
    
    return {
        'title_accuracy': title_accuracy,
        'heading_precision': heading_metrics['precision'],
        'heading_recall': heading_metrics['recall'],
        'heading_f1': heading_metrics['f1_score'],
        'total_files': total_files
    }

if __name__ == "__main__":
    main()
