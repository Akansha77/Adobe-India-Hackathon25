#!/usr/bin/env python3
"""
🏆 Adobe India Hackathon 2025 - Collection 2 F1 Score Calculator
Calculate F1 Score, Precision, and Recall for Collection 2 HR Professional processor

📊 EVALUATION METRICS:
- Precision: How many selected items are relevant
- Recall: How many relevant items are selected  
- F1 Score: Harmonic mean of precision and recall
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse
from difflib import SequenceMatcher


class Collection2F1Calculator:
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Text similarity threshold for matching
        self.similarity_threshold = 0.4
        
        if self.debug:
            print("🔍 Collection 2 F1 Score Calculator initialized")
    
    def calculate_f1_score(self, 
                          generated_file: str = "collection2_output.json",
                          expected_file: str = "Collection 2/challenge1b_output.json") -> Dict[str, float]:
        """
        Calculate F1 score by comparing generated vs expected outputs
        
        Args:
            generated_file: Path to our generated output
            expected_file: Path to the expected/ground truth output
            
        Returns:
            dict: Metrics including precision, recall, and f1_score
        """
        
        # Load both files
        generated_data = self._load_json_file(generated_file)
        expected_data = self._load_json_file(expected_file)
        
        if not generated_data or not expected_data:
            return {"error": "Could not load comparison files"}
        
        # Extract relevant content for comparison
        generated_content = self._extract_comparable_content(generated_data)
        expected_content = self._extract_comparable_content(expected_data)
        
        if self.debug:
            print(f"📊 Generated content items: {len(generated_content)}")
            print(f"📊 Expected content items: {len(expected_content)}")
        
        # Calculate precision and recall
        precision, recall, matches = self._calculate_precision_recall(
            generated_content, expected_content
        )
        
        # Calculate F1 score
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Detailed analysis
        if self.debug:
            self._print_detailed_analysis(generated_content, expected_content, matches)
        
        results = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "generated_items": len(generated_content),
            "expected_items": len(expected_content),
            "matched_items": len(matches)
        }
        
        return results
    
    def _load_json_file(self, file_path: str) -> Dict:
        """Load JSON file safely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            if self.debug:
                print(f"❌ Error loading {file_path}: {e}")
            return {}
    
    def _extract_comparable_content(self, data: Dict) -> List[Dict]:
        """
        Extract comparable content from JSON structure
        Combines both main sections and subsections for comprehensive comparison
        """
        content_items = []
        
        # Extract main sections
        if "extracted_sections" in data:
            for section in data["extracted_sections"]:
                content_items.append({
                    "type": "section",
                    "document": section.get("document", ""),
                    "title": section.get("section_title", ""),
                    "page": section.get("page_number", 0),
                    "text": section.get("section_title", "")
                })
        
        # Extract subsections/refined text
        if "subsection_analysis" in data:
            for subsection in data["subsection_analysis"]:
                content_items.append({
                    "type": "subsection", 
                    "document": subsection.get("document", ""),
                    "title": "",  # subsections don't have titles
                    "page": subsection.get("page_number", 0),
                    "text": subsection.get("refined_text", "")
                })
        
        return content_items
    
    def _calculate_precision_recall(self, 
                                   generated: List[Dict], 
                                   expected: List[Dict]) -> Tuple[float, float, List[Tuple]]:
        """
        Calculate precision and recall using text similarity matching
        
        Returns:
            tuple: (precision, recall, list_of_matches)
        """
        matches = []
        
        # Find matches using text similarity
        for gen_item in generated:
            best_match = None
            best_score = 0.0
            
            for exp_item in expected:
                # Calculate similarity score
                similarity = self._calculate_similarity(gen_item, exp_item)
                
                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_score = similarity
                    best_match = exp_item
            
            if best_match:
                matches.append((gen_item, best_match, best_score))
                if self.debug:
                    print(f"✅ MATCH ({best_score:.3f}): {gen_item['document'][:30]}... -> {best_match['document'][:30]}...")
        
        # Calculate metrics
        precision = len(matches) / len(generated) if generated else 0.0
        recall = len(matches) / len(expected) if expected else 0.0
        
        return precision, recall, matches
    
    def _calculate_similarity(self, item1: Dict, item2: Dict) -> float:
        """
        Calculate similarity between two content items
        Uses multiple factors: document name, page number, text content
        """
        
        # Document name similarity (40% weight)
        doc_similarity = SequenceMatcher(None, 
                                       item1["document"].lower(), 
                                       item2["document"].lower()).ratio()
        
        # Text content similarity (50% weight)
        text1 = item1["text"].lower().strip()
        text2 = item2["text"].lower().strip()
        text_similarity = SequenceMatcher(None, text1, text2).ratio()
        
        # Page number similarity (10% weight)
        page_similarity = 1.0 if item1["page"] == item2["page"] else 0.0
        
        # Weighted combination
        total_similarity = (
            doc_similarity * 0.4 + 
            text_similarity * 0.5 + 
            page_similarity * 0.1
        )
        
        return total_similarity
    
    def _print_detailed_analysis(self, 
                               generated: List[Dict], 
                               expected: List[Dict], 
                               matches: List[Tuple]):
        """Print detailed analysis for debugging"""
        
        print("\n" + "="*80)
        print("📊 DETAILED F1 SCORE ANALYSIS")
        print("="*80)
        
        print(f"\n🎯 MATCHES FOUND ({len(matches)}):")
        for i, (gen, exp, score) in enumerate(matches, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Generated: {gen['document']} (page {gen['page']})")
            print(f"   Expected:  {exp['document']} (page {exp['page']})")
            print(f"   Gen Text:  {gen['text'][:100]}...")
            print(f"   Exp Text:  {exp['text'][:100]}...")
        
        # Show unmatched items
        matched_gen = {id(match[0]) for match in matches}
        matched_exp = {id(match[1]) for match in matches}
        
        unmatched_gen = [item for item in generated if id(item) not in matched_gen]
        unmatched_exp = [item for item in expected if id(item) not in matched_exp]
        
        if unmatched_gen:
            print(f"\n❌ UNMATCHED GENERATED ITEMS ({len(unmatched_gen)}):")
            for i, item in enumerate(unmatched_gen, 1):
                print(f"{i}. {item['document']} (page {item['page']}) - {item['text'][:80]}...")
        
        if unmatched_exp:
            print(f"\n❌ UNMATCHED EXPECTED ITEMS ({len(unmatched_exp)}):")
            for i, item in enumerate(unmatched_exp, 1):
                print(f"{i}. {item['document']} (page {item['page']}) - {item['text'][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Calculate F1 score for Collection 2")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--generated", default="collection2_output.json", 
                       help="Path to generated output file")
    parser.add_argument("--expected", default="Collection 2/challenge1b_output.json",
                       help="Path to expected output file")
    
    args = parser.parse_args()
    
    calculator = Collection2F1Calculator(debug=args.debug)
    
    print("🔍 Calculating F1 Score for Collection 2...")
    results = calculator.calculate_f1_score(args.generated, args.expected)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return 1
    
    print("\n" + "="*60)
    print("📊 COLLECTION 2 EVALUATION RESULTS")
    print("="*60)
    print(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"F1 Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    print("="*60)
    print(f"Generated Items: {results['generated_items']}")
    print(f"Expected Items:  {results['expected_items']}")
    print(f"Matched Items:   {results['matched_items']}")
    print("="*60)
    
    # Interpretation
    if results['f1_score'] >= 0.8:
        print("🏆 EXCELLENT: High quality content extraction!")
    elif results['f1_score'] >= 0.6:
        print("✅ GOOD: Satisfactory content matching")
    elif results['f1_score'] >= 0.4:
        print("⚠️  MODERATE: Some content matching issues")
    else:
        print("❌ POOR: Significant content matching problems")
    
    return 0


if __name__ == "__main__":
    exit(main())
