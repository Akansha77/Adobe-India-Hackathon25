#!/usr/bin/env python3
"""
F1 Score Calculator for Challenge 1B Collection 1
Calculates Precision, Recall, and F1 Score by comparing actual vs expected outputs
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re

def load_json_safe(file_path: str) -> Dict:
    """Safely load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load {file_path}: {e}")
        return {}

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    # Remove extra spaces, special characters, convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_key_phrases(text: str) -> Set[str]:
    """Extract key phrases from text for comparison"""
    normalized = normalize_text(text)
    words = normalized.split()
    
    # Extract meaningful phrases (2-4 word combinations)
    phrases = set()
    
    # Single important words
    for word in words:
        if len(word) > 3:  # Skip very short words
            phrases.add(word)
    
    # Two-word phrases
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if len(phrase) > 6:  # Skip very short phrases
            phrases.add(phrase)
    
    # Three-word phrases for key concepts
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        if len(phrase) > 10:
            phrases.add(phrase)
    
    return phrases

def calculate_section_similarity(actual_section: Dict, expected_section: Dict) -> float:
    """Calculate similarity between two sections"""
    # Check document match
    doc_match = actual_section.get('document', '') == expected_section.get('document', '')
    
    # Check page match  
    page_match = actual_section.get('page_number', 0) == expected_section.get('page_number', 0)
    
    # Extract text content for comparison
    actual_text = actual_section.get('section_title', '') + ' ' + actual_section.get('refined_text', '')
    expected_text = expected_section.get('section_title', '') + ' ' + expected_section.get('refined_text', '')
    
    # Get key phrases
    actual_phrases = extract_key_phrases(actual_text)
    expected_phrases = extract_key_phrases(expected_text)
    
    # Calculate text similarity (Jaccard similarity)
    if not expected_phrases:
        text_similarity = 1.0 if not actual_phrases else 0.0
    else:
        intersection = len(actual_phrases.intersection(expected_phrases))
        union = len(actual_phrases.union(expected_phrases))
        text_similarity = intersection / union if union > 0 else 0.0
    
    # Weighted similarity score
    similarity = 0.0
    if doc_match:
        similarity += 0.3
    if page_match:
        similarity += 0.2
    similarity += text_similarity * 0.5
    
    return similarity

def calculate_f1_score(actual_output: Dict, expected_output: Dict) -> Dict[str, float]:
    """Calculate F1 score for the outputs"""
    
    # Extract sections
    actual_sections = actual_output.get('extracted_sections', [])
    expected_sections = expected_output.get('extracted_sections', [])
    
    actual_subsections = actual_output.get('subsection_analysis', [])
    expected_subsections = expected_output.get('subsection_analysis', [])
    
    # Combine sections for comprehensive analysis
    all_actual = actual_sections + actual_subsections
    all_expected = expected_sections + expected_subsections
    
    print(f"INFO: Comparing {len(all_actual)} actual sections vs {len(all_expected)} expected sections")
    
    # Find matches using similarity threshold
    similarity_threshold = 0.5
    matched_pairs = []
    used_expected = set()
    
    for i, actual_section in enumerate(all_actual):
        best_match = None
        best_similarity = 0.0
        best_j = -1
        
        for j, expected_section in enumerate(all_expected):
            if j in used_expected:
                continue
                
            similarity = calculate_section_similarity(actual_section, expected_section)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = expected_section
                best_j = j
        
        if best_match:
            matched_pairs.append((actual_section, best_match, best_similarity))
            used_expected.add(best_j)
            if len(matched_pairs) <= 5:  # Show first few matches
                print(f"MATCH {len(matched_pairs)}: {best_similarity:.3f} similarity")
                print(f"  Actual: {actual_section.get('section_title', 'N/A')[:60]}...")
                print(f"  Expected: {best_match.get('section_title', 'N/A')[:60]}...")
    
    # Calculate metrics
    true_positives = len(matched_pairs)
    false_positives = len(all_actual) - true_positives
    false_negatives = len(all_expected) - true_positives
    
    # Calculate Precision, Recall, F1
    precision = true_positives / len(all_actual) if len(all_actual) > 0 else 0.0
    recall = true_positives / len(all_expected) if len(all_expected) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate average similarity of matches
    avg_similarity = sum(sim for _, _, sim in matched_pairs) / len(matched_pairs) if matched_pairs else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_actual': len(all_actual),
        'total_expected': len(all_expected),
        'avg_similarity': avg_similarity,
        'matches': matched_pairs
    }

def analyze_metadata_accuracy(actual_output: Dict, expected_output: Dict) -> Dict[str, float]:
    """Analyze metadata field accuracy"""
    actual_meta = actual_output.get('metadata', {})
    expected_meta = expected_output.get('metadata', {})
    
    # Check required fields
    required_fields = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
    field_scores = {}
    
    for field in required_fields:
        if field in actual_meta and field in expected_meta:
            if field == 'input_documents':
                # Compare document lists
                actual_docs = set(actual_meta[field])
                expected_docs = set(expected_meta[field])
                field_scores[field] = len(actual_docs.intersection(expected_docs)) / len(expected_docs) if expected_docs else 0.0
            elif field == 'processing_timestamp':
                # Timestamp just needs to exist
                field_scores[field] = 1.0 if actual_meta[field] else 0.0
            else:
                # Exact match for other fields
                field_scores[field] = 1.0 if actual_meta[field] == expected_meta[field] else 0.0
        else:
            field_scores[field] = 0.0
    
    overall_meta_score = sum(field_scores.values()) / len(required_fields)
    return {
        'overall_score': overall_meta_score,
        'field_scores': field_scores
    }

def main():
    """Main F1 score calculation"""
    print("=" * 60)
    print("🎯 F1 SCORE CALCULATION - Challenge 1B Collection 1")
    print("=" * 60)
    
    # Load files
    actual_output = load_json_safe("collection1_output.json")
    expected_output = load_json_safe("challenge1b_output.json")
    
    if not actual_output or not expected_output:
        print("ERROR: Could not load required files")
        return
    
    # Calculate F1 score for content
    print("\\n📊 CONTENT F1 SCORE ANALYSIS")
    print("-" * 40)
    content_metrics = calculate_f1_score(actual_output, expected_output)
    
    print(f"\\n📈 RESULTS:")
    print(f"True Positives:  {content_metrics['true_positives']}")
    print(f"False Positives: {content_metrics['false_positives']}")
    print(f"False Negatives: {content_metrics['false_negatives']}")
    print(f"\\nPrecision: {content_metrics['precision']:.3f} ({content_metrics['precision']*100:.1f}%)")
    print(f"Recall:    {content_metrics['recall']:.3f} ({content_metrics['recall']*100:.1f}%)")
    print(f"F1 Score:  {content_metrics['f1_score']:.3f} ({content_metrics['f1_score']*100:.1f}%)")
    print(f"Average Match Similarity: {content_metrics['avg_similarity']:.3f}")
    
    # Calculate metadata accuracy
    print("\\n📋 METADATA ACCURACY ANALYSIS")
    print("-" * 40)
    meta_metrics = analyze_metadata_accuracy(actual_output, expected_output)
    
    print(f"Overall Metadata Score: {meta_metrics['overall_score']:.3f} ({meta_metrics['overall_score']*100:.1f}%)")
    print("Field-by-field breakdown:")
    for field, score in meta_metrics['field_scores'].items():
        status = "✅" if score > 0.8 else "⚠️" if score > 0.5 else "❌"
        print(f"  {status} {field}: {score:.3f} ({score*100:.1f}%)")
    
    # Overall assessment
    print("\\n🏆 OVERALL ASSESSMENT")
    print("=" * 40)
    
    # Weighted overall score (70% content, 30% metadata)
    overall_score = (content_metrics['f1_score'] * 0.7) + (meta_metrics['overall_score'] * 0.3)
    
    print(f"Content F1 Score:     {content_metrics['f1_score']:.3f} (70% weight)")
    print(f"Metadata Accuracy:    {meta_metrics['overall_score']:.3f} (30% weight)")
    print(f"\\n🎯 OVERALL F1 SCORE: {overall_score:.3f} ({overall_score*100:.1f}%)")
    
    # Performance classification
    if overall_score >= 0.9:
        grade = "EXCELLENT (A+)"
        emoji = "🏆"
    elif overall_score >= 0.8:
        grade = "VERY GOOD (A)"
        emoji = "⭐"
    elif overall_score >= 0.7:
        grade = "GOOD (B+)"
        emoji = "✅"
    elif overall_score >= 0.6:
        grade = "ACCEPTABLE (B)"
        emoji = "⚠️"
    elif overall_score >= 0.5:
        grade = "NEEDS IMPROVEMENT (C)"
        emoji = "🔧"
    else:
        grade = "POOR (D)"
        emoji = "❌"
    
    print(f"\\n{emoji} GRADE: {grade}")
    
    # Detailed insights
    print("\\n💡 INSIGHTS & RECOMMENDATIONS")
    print("-" * 40)
    
    if content_metrics['precision'] > content_metrics['recall']:
        print("• High precision, lower recall: You're finding good matches but missing some expected content")
        print("• Recommendation: Broaden section extraction criteria")
    elif content_metrics['recall'] > content_metrics['precision']:
        print("• High recall, lower precision: You're capturing most content but with some noise")
        print("• Recommendation: Improve filtering to focus on most relevant content")
    else:
        print("• Balanced precision and recall")
    
    if content_metrics['avg_similarity'] < 0.7:
        print("• Low average similarity suggests content quality differences")
        print("• Recommendation: Improve text extraction and section identification")
    
    print(f"\\n📊 SUMMARY STATISTICS:")
    print(f"• Sections Found: {content_metrics['total_actual']}")
    print(f"• Sections Expected: {content_metrics['total_expected']}")
    print(f"• Successful Matches: {content_metrics['true_positives']}")
    print(f"• Match Rate: {content_metrics['true_positives']/content_metrics['total_expected']*100:.1f}%")

if __name__ == "__main__":
    main()
