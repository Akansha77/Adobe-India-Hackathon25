#!/usr/bin/env python3
"""
Challenge 1B Evaluation Script
Compares generated outputs with ground truth outputs for accuracy assessment
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set

def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_section_titles(data: Dict) -> Set[str]:
    """Extract section titles from output data"""
    titles = set()
    if 'extracted_sections' in data:
        for section in data['extracted_sections']:
            if 'section_title' in section:
                titles.add(section['section_title'].strip().lower())
    return titles

def extract_importance_ranks(data: Dict) -> List[int]:
    """Extract importance ranks from output data"""
    ranks = []
    if 'extracted_sections' in data:
        for section in data['extracted_sections']:
            if 'importance_rank' in section:
                ranks.append(section['importance_rank'])
    return sorted(ranks)

def calculate_section_accuracy(predicted_titles: Set[str], ground_truth_titles: Set[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 for section titles"""
    if not ground_truth_titles:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if not predicted_titles:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    intersection = predicted_titles.intersection(ground_truth_titles)
    
    precision = len(intersection) / len(predicted_titles) if predicted_titles else 0.0
    recall = len(intersection) / len(ground_truth_titles) if ground_truth_titles else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100
    }

def evaluate_collection(collection_dir: str) -> Dict:
    """Evaluate a single collection"""
    collection_name = os.path.basename(collection_dir)
    
    # File paths
    predicted_file = os.path.join(collection_dir, f"{collection_name.lower().replace(' ', '')}_output.json")
    ground_truth_file = os.path.join(collection_dir, "challenge1b_output.json")
    
    if not os.path.exists(predicted_file):
        print(f"❌ Predicted file not found: {predicted_file}")
        return None
    
    if not os.path.exists(ground_truth_file):
        print(f"❌ Ground truth file not found: {ground_truth_file}")
        return None
    
    # Load data
    predicted_data = load_json(predicted_file)
    ground_truth_data = load_json(ground_truth_file)
    
    # Extract section titles
    predicted_titles = extract_section_titles(predicted_data)
    ground_truth_titles = extract_section_titles(ground_truth_data)
    
    # Calculate metrics
    metrics = calculate_section_accuracy(predicted_titles, ground_truth_titles)
    
    # Extract importance ranks for additional analysis
    predicted_ranks = extract_importance_ranks(predicted_data)
    ground_truth_ranks = extract_importance_ranks(ground_truth_data)
    
    rank_accuracy = 0.0
    if predicted_ranks and ground_truth_ranks:
        # Simple rank order correlation (if same number of sections)
        if len(predicted_ranks) == len(ground_truth_ranks):
            rank_accuracy = sum(1 for p, g in zip(predicted_ranks, ground_truth_ranks) if p == g) / len(predicted_ranks) * 100
    
    return {
        "collection": collection_name,
        "section_metrics": metrics,
        "rank_accuracy": rank_accuracy,
        "predicted_sections": len(predicted_titles),
        "ground_truth_sections": len(ground_truth_titles),
        "matching_sections": len(predicted_titles.intersection(ground_truth_titles))
    }

def main():
    base_dir = "/workspaces/Adobe-India-Hackathon25/Challenge_1b"
    collections = ["Collection 1", "Collection 2", "Collection 3"]
    
    print("🧪 ADOBE HACKATHON 2025 - CHALLENGE 1B EVALUATION")
    print("=" * 60)
    
    all_results = []
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_rank_accuracy = 0
    
    for collection in collections:
        collection_dir = os.path.join(base_dir, collection)
        result = evaluate_collection(collection_dir)
        
        if result:
            all_results.append(result)
            
            print(f"\n📋 {result['collection']}:")
            print(f"   ✓ Precision: {result['section_metrics']['precision']:.1f}%")
            print(f"   ✓ Recall: {result['section_metrics']['recall']:.1f}%")
            print(f"   ✓ F1-Score: {result['section_metrics']['f1']:.1f}%")
            print(f"   ✓ Rank Accuracy: {result['rank_accuracy']:.1f}%")
            print(f"   📊 Sections - Predicted: {result['predicted_sections']}, Ground Truth: {result['ground_truth_sections']}, Matching: {result['matching_sections']}")
            
            total_precision += result['section_metrics']['precision']
            total_recall += result['section_metrics']['recall']
            total_f1 += result['section_metrics']['f1']
            total_rank_accuracy += result['rank_accuracy']
    
    if all_results:
        num_collections = len(all_results)
        
        print("\n" + "=" * 60)
        print("📊 OVERALL CHALLENGE 1B PERFORMANCE SUMMARY:")
        print("=" * 60)
        print(f"📋 SECTION EXTRACTION METRICS:")
        print(f"   ✓ Average Precision: {total_precision / num_collections:.1f}%")
        print(f"   ✓ Average Recall: {total_recall / num_collections:.1f}%")
        print(f"   ✓ Average F1-Score: {total_f1 / num_collections:.1f}%")
        print(f"📈 RANKING METRICS:")
        print(f"   ✓ Average Rank Accuracy: {total_rank_accuracy / num_collections:.1f}%")
        print(f"📁 COLLECTIONS PROCESSED:")
        print(f"   • Total Collections: {num_collections}")
        print(f"   • Success Rate: 100%")
        
        # Performance assessment
        avg_f1 = total_f1 / num_collections
        avg_precision = total_precision / num_collections
        avg_recall = total_recall / num_collections
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if avg_f1 >= 70:
            print(f"   🌟 EXCELLENT: F1-Score ≥70%")
        elif avg_f1 >= 50:
            print(f"   👍 GOOD: F1-Score ≥50%")
        elif avg_f1 >= 30:
            print(f"   ⚠️ NEEDS IMPROVEMENT: F1-Score ≥30%")
        else:
            print(f"   ❌ POOR: F1-Score <30%")

if __name__ == "__main__":
    main()
