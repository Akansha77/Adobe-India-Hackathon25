import json
import os
from typing import Dict, List, Tuple

def calculate_metrics():
    """Calculate precision, recall, accuracy, and F1-score for PDF processing"""
    
    expected_files = ['file01.json', 'file02.json', 'file03.json', 'file04.json', 'file05.json']
    expected_dir = 'sample_dataset/outputs'
    actual_dir = 'test_output'
    
    print("DETAILED ACCURACY & PRECISION ANALYSIS")
    print("=" * 50)
    
    # Overall metrics
    total_title_correct = 0
    total_title_count = 0
    total_outline_tp = 0  # True Positives
    total_outline_fp = 0  # False Positives  
    total_outline_fn = 0  # False Negatives
    total_outline_tn = 0  # True Negatives
    
    file_results = []
    
    for filename in expected_files:
        expected_path = os.path.join(expected_dir, filename)
        actual_path = os.path.join(actual_dir, filename)
        
        with open(expected_path, 'r', encoding='utf-8') as f:
            expected = json.load(f)
        with open(actual_path, 'r', encoding='utf-8') as f:
            actual = json.load(f)
        
        # Title Analysis
        title_correct = expected['title'] == actual['title']
        total_title_correct += title_correct
        total_title_count += 1
        
        # Outline Analysis - detailed matching
        expected_outline = expected['outline']
        actual_outline = actual['outline']
        
        # Create sets of (text, level) for comparison
        expected_items = {(item['text'], item['level']) for item in expected_outline}
        actual_items = {(item['text'], item['level']) for item in actual_outline}
        
        # Calculate confusion matrix components
        tp = len(expected_items & actual_items)  # Correctly identified headings
        fp = len(actual_items - expected_items)  # Incorrectly identified as headings
        fn = len(expected_items - actual_items)  # Missed headings
        
        # For outline, TN is not meaningful (we don't count non-headings correctly rejected)
        
        total_outline_tp += tp
        total_outline_fp += fp
        total_outline_fn += fn
        
        # Calculate per-file metrics
        outline_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        outline_recall = tp / (tp + fn) if (tp + fn) > 0 else (1 if tp == 0 and fn == 0 else 0)
        outline_f1 = 2 * (outline_precision * outline_recall) / (outline_precision + outline_recall) if (outline_precision + outline_recall) > 0 else 0
        
        file_results.append({
            'filename': filename,
            'title_correct': title_correct,
            'expected_headings': len(expected_outline),
            'actual_headings': len(actual_outline),
            'tp': tp,
            'fp': fp, 
            'fn': fn,
            'precision': outline_precision,
            'recall': outline_recall,
            'f1': outline_f1
        })
        
        print(f"\n{filename}:")
        print(f"  Title: {'✓' if title_correct else '✗'}")
        print(f"  Expected headings: {len(expected_outline)}")
        print(f"  Actual headings: {len(actual_outline)}")
        print(f"  Correct matches (TP): {tp}")
        print(f"  False positives (FP): {fp}")
        print(f"  False negatives (FN): {fn}")
        print(f"  Outline Precision: {outline_precision:.3f}")
        print(f"  Outline Recall: {outline_recall:.3f}")
        print(f"  Outline F1-Score: {outline_f1:.3f}")
        
        # Show mismatches for debugging
        if fp > 0:
            print(f"  False Positives: {list(actual_items - expected_items)}")
        if fn > 0:
            print(f"  False Negatives: {list(expected_items - actual_items)}")
    
    # Overall metrics calculation
    print(f"\n{'='*50}")
    print("OVERALL METRICS:")
    print("=" * 50)
    
    # Title Accuracy
    title_accuracy = total_title_correct / total_title_count
    print(f"Title Accuracy: {title_accuracy:.3f} ({total_title_correct}/{total_title_count})")
    
    # Outline Metrics
    overall_precision = total_outline_tp / (total_outline_tp + total_outline_fp) if (total_outline_tp + total_outline_fp) > 0 else 0
    overall_recall = total_outline_tp / (total_outline_tp + total_outline_fn) if (total_outline_tp + total_outline_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"Outline Precision: {overall_precision:.3f}")
    print(f"Outline Recall: {overall_recall:.3f}")
    print(f"Outline F1-Score: {overall_f1:.3f}")
    
    # Combined accuracy (title + outline)
    total_correct_items = total_title_correct + total_outline_tp
    total_possible_items = total_title_count + total_outline_tp + total_outline_fn
    combined_accuracy = total_correct_items / total_possible_items if total_possible_items > 0 else 0
    
    print(f"\nCombined Accuracy: {combined_accuracy:.3f}")
    print(f"  Correct items: {total_correct_items}")
    print(f"  Total possible: {total_possible_items}")
    
    # Performance summary
    print(f"\n{'='*50}")
    print("PERFORMANCE SUMMARY:")
    print("=" * 50)
    print(f"• Total True Positives: {total_outline_tp}")
    print(f"• Total False Positives: {total_outline_fp}")
    print(f"• Total False Negatives: {total_outline_fn}")
    print(f"• Model correctly identified {total_outline_tp} out of {total_outline_tp + total_outline_fn} expected headings")
    print(f"• Model incorrectly identified {total_outline_fp} non-headings as headings")
    
    # Quality assessment
    if overall_precision >= 0.9:
        precision_grade = "Excellent"
    elif overall_precision >= 0.8:
        precision_grade = "Good"
    elif overall_precision >= 0.7:
        precision_grade = "Fair"
    else:
        precision_grade = "Needs Improvement"
    
    if overall_recall >= 0.9:
        recall_grade = "Excellent"
    elif overall_recall >= 0.8:
        recall_grade = "Good"
    elif overall_recall >= 0.7:
        recall_grade = "Fair"
    else:
        recall_grade = "Needs Improvement"
    
    print(f"\nQuality Assessment:")
    print(f"• Precision: {precision_grade} ({overall_precision:.1%})")
    print(f"• Recall: {recall_grade} ({overall_recall:.1%})")
    print(f"• Overall Performance: {'Excellent' if overall_f1 >= 0.9 else 'Good' if overall_f1 >= 0.8 else 'Fair' if overall_f1 >= 0.7 else 'Needs Improvement'}")

if __name__ == "__main__":
    calculate_metrics()
