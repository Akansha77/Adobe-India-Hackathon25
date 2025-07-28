#!/usr/bin/env python3
"""
Evaluation Script for PDF Processing Results
Calculates F1-score, Precision, and Recall for title and heading extraction
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import difflib

class PDFEvaluator:
    """Evaluates PDF processing results against ground truth"""
    
    def __init__(self):
        self.results = {
            'title_matches': 0,
            'total_files': 0,
            'total_predicted_headings': 0,
            'total_ground_truth_headings': 0,
            'correct_headings': 0,
            'file_results': []
        }
    
    def evaluate(self, predicted_dir: Path, ground_truth_dir: Path) -> Dict:
        """
        Evaluate predicted results against ground truth
        
        Args:
            predicted_dir: Directory with predicted JSON files
            ground_truth_dir: Directory with ground truth JSON files
            
        Returns:
            Dict with evaluation metrics
        """
        print("🧪 ADOBE HACKATHON 2025 - PERFORMANCE EVALUATION")
        print("=" * 60)
        
        # Find common files
        predicted_files = {f.stem: f for f in predicted_dir.glob("*.json")}
        ground_truth_files = {f.stem: f for f in ground_truth_dir.glob("*.json")}
        
        common_files = set(predicted_files.keys()) & set(ground_truth_files.keys())
        
        if not common_files:
            print("❌ No common files found between predicted and ground truth")
            return {}
        
        self.results['total_files'] = len(common_files)
        
        # Evaluate titles
        print("🏷️  TITLE EVALUATION:")
        print("-" * 40)
        
        for file_name in sorted(common_files):
            title_match = self._evaluate_title(
                predicted_files[file_name],
                ground_truth_files[file_name]
            )
            print(f"{file_name}.json  | {'✅' if title_match else '❌'} | Match: {title_match}")
        
        # Evaluate headings
        print("\n📋 HEADING EVALUATION:")
        print("-" * 40)
        
        for file_name in sorted(common_files):
            file_result = self._evaluate_headings(
                predicted_files[file_name],
                ground_truth_files[file_name],
                file_name
            )
            self.results['file_results'].append(file_result)
            
            precision = (file_result['correct'] / file_result['predicted']) * 100 if file_result['predicted'] > 0 else 0
            recall = (file_result['correct'] / file_result['ground_truth']) * 100 if file_result['ground_truth'] > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{file_name}.json  | P: {file_result['predicted']:2d} | GT: {file_result['ground_truth']:2d} | C: {file_result['correct']:2d} | Prec: {precision:.1f}% | Rec: {recall:.1f}% | F1: {f1:.1f}%")
        
        # Calculate overall metrics
        return self._calculate_overall_metrics()
    
    def _evaluate_title(self, predicted_file: Path, ground_truth_file: Path) -> bool:
        """Evaluate title extraction"""
        try:
            with open(predicted_file, 'r', encoding='utf-8') as f:
                predicted = json.load(f)
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            predicted_title = predicted.get('title', '').strip()
            ground_truth_title = ground_truth.get('title', '').strip()
            
            # Normalize for comparison
            predicted_normalized = self._normalize_text(predicted_title)
            ground_truth_normalized = self._normalize_text(ground_truth_title)
            
            match = predicted_normalized == ground_truth_normalized
            if match:
                self.results['title_matches'] += 1
            
            return match
            
        except Exception as e:
            print(f"Error evaluating title for {predicted_file.name}: {e}")
            return False
    
    def _evaluate_headings(self, predicted_file: Path, ground_truth_file: Path, file_name: str) -> Dict:
        """Evaluate heading extraction"""
        try:
            with open(predicted_file, 'r', encoding='utf-8') as f:
                predicted = json.load(f)
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            predicted_headings = predicted.get('outline', [])
            ground_truth_headings = ground_truth.get('outline', [])
            
            # Extract heading texts for comparison
            predicted_texts = set()
            for heading in predicted_headings:
                text = self._normalize_text(heading.get('text', ''))
                if text:
                    predicted_texts.add(text)
            
            ground_truth_texts = set()
            for heading in ground_truth_headings:
                text = self._normalize_text(heading.get('text', ''))
                if text:
                    ground_truth_texts.add(text)
            
            # Calculate matches
            correct_headings = len(predicted_texts & ground_truth_texts)
            
            # Update totals
            self.results['total_predicted_headings'] += len(predicted_texts)
            self.results['total_ground_truth_headings'] += len(ground_truth_texts)
            self.results['correct_headings'] += correct_headings
            
            return {
                'predicted': len(predicted_texts),
                'ground_truth': len(ground_truth_texts),
                'correct': correct_headings
            }
            
        except Exception as e:
            print(f"Error evaluating headings for {predicted_file.name}: {e}")
            return {'predicted': 0, 'ground_truth': 0, 'correct': 0}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        normalized = ' '.join(text.split()).strip()
        
        # Remove trailing spaces that might be added for consistency
        normalized = normalized.rstrip()
        
        return normalized.lower()
    
    def _calculate_overall_metrics(self) -> Dict:
        """Calculate overall performance metrics"""
        print("\n" + "=" * 60)
        print("📊 OVERALL PERFORMANCE SUMMARY:")
        print("=" * 60)
        
        # Title metrics
        title_accuracy = (self.results['title_matches'] / self.results['total_files']) * 100
        print("🏷️  TITLE METRICS:")
        print(f"   ✓ Accuracy: {title_accuracy:.1f}% ({self.results['title_matches']}/{self.results['total_files']})")
        
        # Heading metrics
        total_predicted = self.results['total_predicted_headings']
        total_ground_truth = self.results['total_ground_truth_headings']
        total_correct = self.results['correct_headings']
        
        precision = (total_correct / total_predicted) * 100 if total_predicted > 0 else 0
        recall = (total_correct / total_ground_truth) * 100 if total_ground_truth > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("📋 HEADING METRICS:")
        print(f"   ✓ Precision: {precision:.1f}%")
        print(f"   ✓ Recall: {recall:.1f}%")
        print(f"   ✓ F1-Score: {f1_score:.1f}%")
        
        print("📁 DETAILED COUNTS:")
        print(f"   • Files Processed: {self.results['total_files']}")
        print(f"   • Total Predicted Headings: {total_predicted}")
        print(f"   • Total Ground Truth Headings: {total_ground_truth}")
        print(f"   • Correctly Matched Headings: {total_correct}")
        
        # Performance assessment
        print("🎯 PERFORMANCE ASSESSMENT:")
        if title_accuracy >= 90:
            print("   🏆 EXCELLENT: Title accuracy ≥90%")
        elif title_accuracy >= 70:
            print("   👍 GOOD: Title accuracy ≥70%")
        else:
            print("   ⚠️  NEEDS IMPROVEMENT: Title accuracy <70%")
        
        if f1_score >= 60:
            print("   🏆 EXCELLENT: Heading F1-score ≥60%")
        elif f1_score >= 40:
            print("   👍 GOOD: Heading F1-score ≥40%")
        elif f1_score >= 20:
            print("   ⚠️  FAIR: Heading F1-score ≥20%")
        else:
            print("   ❌ POOR: Heading F1-score <20%")
        
        return {
            'title_accuracy': title_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_files': self.results['total_files'],
            'title_matches': self.results['title_matches'],
            'total_predicted_headings': total_predicted,
            'total_ground_truth_headings': total_ground_truth,
            'correct_headings': total_correct
        }

def main():
    """Main evaluation function"""
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <predicted_dir> <ground_truth_dir>")
        print("Example: python evaluate.py test_outputs sample_dataset/outputs")
        sys.exit(1)
    
    predicted_dir = Path(sys.argv[1])
    ground_truth_dir = Path(sys.argv[2])
    
    if not predicted_dir.exists():
        print(f"❌ Predicted directory not found: {predicted_dir}")
        sys.exit(1)
    
    if not ground_truth_dir.exists():
        print(f"❌ Ground truth directory not found: {ground_truth_dir}")
        sys.exit(1)
    
    evaluator = PDFEvaluator()
    results = evaluator.evaluate(predicted_dir, ground_truth_dir)
    
    return results

if __name__ == "__main__":
    main()
