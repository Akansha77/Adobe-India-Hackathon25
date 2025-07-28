#!/usr/bin/env python3
"""
🏆 Adobe India Hackathon 2025 - Challenge 1B Collection 3 Processor
Food Contractor - Vegetarian Buffet Menu Planning

🎯 FOCUSED ON COLLECTION 3:
- Food contractor persona
- Recipe and cooking documents
- Vegetarian buffet-style dinner menu
- Corporate gathering with gluten-free options

🚀 CAPABILITIES:
- Context-aware recipe extraction for buffet planning
- Vegetarian and gluten-free content filtering
- Ingredient and preparation content ranking
- Corporate catering-specific content prioritization
"""

import fitz  # PyMuPDF
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import argparse
import time
import datetime
from dataclasses import dataclass


@dataclass
class DocumentSection:
    """Represents an extracted document section"""
    document: str
    section_title: str
    content: str
    page_number: int
    importance_score: float
    relevance_score: float
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]


@dataclass
class PersonaContext:
    """Context for Food Contractor persona"""
    role: str
    task: str
    constraints: List[str]
    priorities: List[str]


class Collection3Processor:
    """Specialized processor for Collection 3 - Food Contractor buffet menu planning"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Food contractor and buffet catering focused keywords
        self.keyword_config = {
            'high_priority_keywords': [
                # Vegetarian Focus
                'vegetarian', 'vegan', 'plant-based', 'meat-free', 'veggie',
                'vegetables', 'beans', 'lentils', 'chickpeas', 'tofu', 'quinoa',
                
                # Gluten-Free Focus
                'gluten-free', 'gluten free', 'celiac', 'wheat-free', 'no gluten',
                'rice', 'corn', 'potato', 'naturally gluten-free',
                
                # Buffet & Catering
                'buffet', 'catering', 'corporate', 'gathering', 'party', 'event',
                'large batch', 'serves', 'portions', 'crowd', 'multiple servings',
                
                # Recipe Components
                'ingredients', 'recipe', 'preparation', 'cooking', 'instructions',
                'serves', 'prep time', 'cook time', 'method', 'steps',
                
                # Dinner Focus
                'dinner', 'main', 'side', 'sides', 'main course', 'side dish',
                'appetizer', 'salad', 'soup', 'entree', 'meal',
                
                # Cooking Methods
                'baked', 'roasted', 'grilled', 'steamed', 'sauteed', 'stir-fry',
                'boiled', 'braised', 'slow cook', 'oven', 'stovetop',
                
                # Dietary Considerations
                'dairy-free', 'nut-free', 'soy-free', 'healthy', 'nutritious',
                'allergen', 'dietary', 'restriction', 'special diet',
                
                # Food Categories
                'pasta', 'rice', 'grain', 'salad', 'vegetable', 'protein',
                'sauce', 'dressing', 'seasoning', 'herb', 'spice'
            ],
            'importance_weights': {
                'vegetarian_buffet': 1.0,
                'gluten_free': 0.95,
                'dinner_items': 0.9,
                'ingredients': 0.85,
                'preparation': 0.8,
                'catering_suitable': 0.75,
                'dietary_info': 0.7,
                'cooking_methods': 0.65
            }
        }
        
        if self.debug:
            print("SUCCESS: Challenge 1B Collection 3 Processor initialized for Food Contractor")
    
    def process_collection_3(self, collection_path: Path = None) -> Dict[str, Any]:
        """
        Process Collection 3 specifically - Food contractor buffet menu planning
        
        Args:
            collection_path: Path to Collection 3 directory (optional)
            
        Returns:
            dict: Complete analysis results for buffet menu planning
        """
        if collection_path is None:
            collection_path = Path(".")
        
        if self.debug:
            print(f"🔍 Processing Collection 3: {collection_path}")
        
        # Load input configuration dynamically
        input_config = self._load_input_config(collection_path)
        if not input_config:
            return {"error": "Could not load input configuration"}
        
        # Extract persona context
        persona_context = self._extract_persona_context(input_config)
        
        # Process PDF documents
        pdf_directory = collection_path / "PDFs"
        processed_docs = []
        all_sections = []
        
        if pdf_directory.exists():
            pdf_files = list(pdf_directory.glob("*.pdf"))
            if self.debug:
                print(f"📄 Found {len(pdf_files)} PDF documents")
            
            for pdf_file in pdf_files:
                if self.debug:
                    print(f"📖 Processing: {pdf_file.name}")
                
                sections = self._extract_document_sections(pdf_file, persona_context)
                all_sections.extend(sections)
                processed_docs.append(pdf_file.name)
        
        # Rank and select top sections for buffet menu planning
        top_sections = self._rank_sections_for_catering(all_sections, persona_context)
        
        # Generate comprehensive analysis
        analysis = self._generate_catering_analysis(top_sections, persona_context)
        
        # Create final output structure
        output = {
            "metadata": {
                "input_documents": processed_docs,
                "persona": persona_context.role,
                "job_to_be_done": persona_context.task,
                "processing_timestamp": datetime.datetime.now().isoformat()
            },
            "extracted_sections": self._format_sections(top_sections[:5]),  # Top 5 main sections
            "subsection_analysis": self._format_subsections(top_sections[5:])  # Additional detailed sections
        }
        
        return output
    
    def _load_input_config(self, collection_path: Path) -> Dict[str, Any]:
        """Load input configuration from challenge1b_input.json"""
        input_file = collection_path / "challenge1b_input.json"
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            if self.debug:
                print(f"ERROR: Could not load {input_file}: {e}")
            return {}
    
    def _extract_persona_context(self, input_config: Dict[str, Any]) -> PersonaContext:
        """Extract persona context from input configuration"""
        persona = input_config.get("persona", {})
        job_to_be_done = input_config.get("job_to_be_done", {})
        
        # Handle both dict and string formats
        if isinstance(job_to_be_done, dict):
            task = job_to_be_done.get("task", "")
        else:
            task = str(job_to_be_done)
        
        return PersonaContext(
            role=persona.get("role", "Food Contractor"),
            task=task,
            constraints=input_config.get("constraints", []),
            priorities=["Vegetarian buffet", "Gluten-free options", "Corporate catering", "Large batches"]
        )
    
    def _extract_document_sections(self, pdf_path: Path, context: PersonaContext) -> List[DocumentSection]:
        """Extract sections using optimized approach for food contractor buffet planning"""
        sections = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text blocks with positioning
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    # Extract text content
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                    
                    block_text = block_text.strip()
                    if len(block_text) < 10:  # Skip very short text
                        continue
                    
                    # Clean and normalize text
                    cleaned_text = self._clean_text(block_text)
                    if not cleaned_text:
                        continue
                    
                    # Calculate relevance scores
                    importance_score = self._calculate_food_importance(cleaned_text, context)
                    relevance_score = self._calculate_buffet_relevance(cleaned_text, context)
                    
                    # Skip if not relevant enough
                    if importance_score < 0.1 and relevance_score < 0.1:
                        continue
                    
                    # Generate semantic title
                    section_title = self._generate_food_semantic_title(cleaned_text, context)
                    
                    section = DocumentSection(
                        document=pdf_path.name,
                        section_title=section_title,
                        content=cleaned_text,
                        page_number=page_num + 1,
                        importance_score=importance_score,
                        relevance_score=relevance_score,
                        start_pos=(block["bbox"][0], block["bbox"][1]),
                        end_pos=(block["bbox"][2], block["bbox"][3])
                    )
                    
                    sections.append(section)
            
            doc.close()
            
        except Exception as e:
            if self.debug:
                print(f"ERROR: Could not process {pdf_path}: {e}")
        
        return sections
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def _calculate_food_importance(self, text: str, context: PersonaContext) -> float:
        """Calculate importance score for food contractor buffet planning"""
        text_lower = text.lower()
        score = 0.0
        
        # High-value food and catering keywords
        vegetarian_terms = ['vegetarian', 'vegan', 'plant-based', 'meat-free', 'veggie']
        gluten_free_terms = ['gluten-free', 'gluten free', 'celiac', 'wheat-free']
        buffet_terms = ['buffet', 'catering', 'corporate', 'large batch', 'crowd']
        recipe_terms = ['ingredients', 'recipe', 'preparation', 'cooking', 'serves']
        dinner_terms = ['dinner', 'main', 'side', 'main course', 'side dish']
        
        # Score based on keyword presence
        for term in vegetarian_terms:
            if term in text_lower:
                score += 0.2
        
        for term in gluten_free_terms:
            if term in text_lower:
                score += 0.18
        
        for term in buffet_terms:
            if term in text_lower:
                score += 0.15
        
        for term in recipe_terms:
            if term in text_lower:
                score += 0.12
        
        for term in dinner_terms:
            if term in text_lower:
                score += 0.1
        
        # Bonus for recipe structure indicators
        if re.search(r'\d+\s*(cups?|tbsp|tsp|oz|lbs?|servings?)', text_lower):
            score += 0.15  # Contains measurements
        
        if re.search(r'(step|instructions?|method|directions?)', text_lower):
            score += 0.1  # Contains cooking instructions
        
        return min(score, 1.0)
    
    def _calculate_buffet_relevance(self, text: str, context: PersonaContext) -> float:
        """Calculate relevance score for buffet menu context"""
        text_lower = text.lower()
        relevance = 0.0
        
        # Context-specific relevance factors
        task_lower = context.task.lower()
        
        # Match against job requirements
        if 'vegetarian' in task_lower and any(term in text_lower for term in ['vegetarian', 'vegan', 'plant']):
            relevance += 0.3
        
        if 'buffet' in task_lower and any(term in text_lower for term in ['buffet', 'large', 'batch', 'crowd']):
            relevance += 0.25
        
        if 'gluten-free' in task_lower and any(term in text_lower for term in ['gluten-free', 'celiac', 'wheat-free']):
            relevance += 0.2
        
        if 'dinner' in task_lower and any(term in text_lower for term in ['dinner', 'main', 'side']):
            relevance += 0.15
        
        if 'corporate' in task_lower and any(term in text_lower for term in ['professional', 'elegant', 'presentation']):
            relevance += 0.1
        
        return min(relevance, 1.0)
    
    def _classify_content_theme(self, text: str) -> str:
        """Classify content into food contractor themes"""
        text_lower = text.lower()
        
        # Theme classification for food contractor work
        if any(term in text_lower for term in ['vegetarian', 'vegan', 'plant-based', 'meat-free']):
            return 'vegetarian_options'
        elif any(term in text_lower for term in ['gluten-free', 'celiac', 'wheat-free']):
            return 'gluten_free_items'
        elif any(term in text_lower for term in ['ingredients', 'recipe', 'cooking']):
            return 'recipe_details'
        elif any(term in text_lower for term in ['main', 'entree', 'main course']):
            return 'main_dishes'
        elif any(term in text_lower for term in ['side', 'side dish', 'sides']):
            return 'side_dishes'
        elif any(term in text_lower for term in ['buffet', 'catering', 'large batch']):
            return 'buffet_planning'
        elif any(term in text_lower for term in ['preparation', 'prep', 'instructions']):
            return 'cooking_methods'
        else:
            return 'general_food_info'
    
    def _generate_food_semantic_title(self, text: str, context: PersonaContext) -> str:
        """Generate semantic titles for food contractor content"""
        text_lower = text.lower()
        
        # Generate contextual titles based on content
        if 'ingredients' in text_lower and ('vegetarian' in text_lower or 'plant' in text_lower):
            return "Vegetarian ingredients and components"
        elif 'gluten-free' in text_lower and 'recipe' in text_lower:
            return "Gluten-free recipe instructions"
        elif 'buffet' in text_lower or 'large batch' in text_lower:
            return "Large-scale buffet preparation"
        elif 'main' in text_lower and ('dinner' in text_lower or 'entree' in text_lower):
            return "Main course dinner options"
        elif 'side' in text_lower and 'dish' in text_lower:
            return "Side dish preparations"
        elif 'ingredients' in text_lower:
            return "Recipe ingredients list"
        elif 'preparation' in text_lower or 'cooking' in text_lower:
            return "Cooking preparation methods"
        elif 'serves' in text_lower or 'portions' in text_lower:
            return "Serving size and portions"
        else:
            # Generate title from first meaningful phrase
            sentences = re.split(r'[.!?]+', text)
            if sentences:
                first_sentence = sentences[0].strip()[:50]
                return first_sentence if first_sentence else "Food preparation details"
            return "Recipe and cooking information"
    
    def _rank_sections_for_catering(self, sections: List[DocumentSection], context: PersonaContext) -> List[DocumentSection]:
        """Rank sections specifically for food contractor buffet catering needs"""
        
        def catering_score(section: DocumentSection) -> float:
            base_score = (section.importance_score * 0.6) + (section.relevance_score * 0.4)
            
            # Boost for catering-specific content
            content_lower = section.content.lower()
            
            # Higher priority content
            if any(term in content_lower for term in ['vegetarian', 'vegan', 'plant-based']):
                base_score *= 1.3
            
            if any(term in content_lower for term in ['gluten-free', 'celiac']):
                base_score *= 1.25
            
            if any(term in content_lower for term in ['buffet', 'large batch', 'serves many']):
                base_score *= 1.2
            
            if 'ingredients' in content_lower:
                base_score *= 1.15
            
            if any(term in content_lower for term in ['dinner', 'main course', 'side dish']):
                base_score *= 1.1
            
            return base_score
        
        # Sort by catering relevance score
        ranked_sections = sorted(sections, key=catering_score, reverse=True)
        
        if self.debug:
            print(f"📊 Ranked {len(ranked_sections)} sections for food contractor catering")
        
        return ranked_sections
    
    def _generate_catering_analysis(self, sections: List[DocumentSection], context: PersonaContext) -> Dict[str, Any]:
        """Generate comprehensive analysis for food contractor buffet planning"""
        
        analysis = {
            "total_sections": len(sections),
            "vegetarian_content": len([s for s in sections if 'vegetarian' in s.content.lower()]),
            "gluten_free_content": len([s for s in sections if 'gluten-free' in s.content.lower()]),
            "buffet_suitable": len([s for s in sections if any(term in s.content.lower() 
                                   for term in ['buffet', 'large batch', 'catering'])]),
            "recipe_sections": len([s for s in sections if 'ingredients' in s.content.lower()]),
            "themes": Counter([self._classify_content_theme(s.content) for s in sections])
        }
        
        if self.debug:
            print(f"📊 Catering Analysis: {analysis['vegetarian_content']} vegetarian, "
                  f"{analysis['gluten_free_content']} gluten-free, "
                  f"{analysis['buffet_suitable']} buffet-suitable sections")
        
        return analysis
    
    def _format_sections(self, sections: List[DocumentSection]) -> List[Dict[str, Any]]:
        """Format main sections for output"""
        return [
            {
                "document": section.document,
                "section_title": section.section_title,
                "importance_rank": i + 1,
                "page_number": section.page_number
            }
            for i, section in enumerate(sections)
        ]
    
    def _format_subsections(self, sections: List[DocumentSection]) -> List[Dict[str, Any]]:
        """Format subsections for detailed analysis"""
        return [
            {
                "document": section.document,
                "refined_text": section.content[:200] + "..." if len(section.content) > 200 else section.content,
                "page_number": section.page_number
            }
            for section in sections
        ]


def main():
    parser = argparse.ArgumentParser(description="Process Collection 3 for Food Contractor buffet planning")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--output", default="collection3_output.json", 
                       help="Output file name")
    
    args = parser.parse_args()
    
    processor = Collection3Processor(debug=args.debug)
    
    if args.debug:
        print("Starting Collection 3 processing for Food Contractor...")
    
    try:
        # Process Collection 3
        results = processor.process_collection_3()
        
        if "error" in results:
            print(f"❌ ERROR: {results['error']}")
            return 1
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if args.debug:
            print(f"✅ SUCCESS: Collection 3 analysis completed!")
            print(f"📄 Generated: {args.output}")
            
            # Print summary
            sections_count = len(results.get("extracted_sections", []))
            subsections_count = len(results.get("subsection_analysis", []))
            print(f"📊 SUMMARY: {sections_count} main sections, {subsections_count} detailed subsections")
        
    except Exception as e:
        print(f"❌ ERROR: Could not save output file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
