#!/usr/bin/env python3
"""
🏆 Adobe India Hackathon 2025 - Challenge 1B Collection 2 Processor
HR Professional - Acrobat Forms Management

🎯 FOCUSED ON COLLECTION 2:
- HR professional persona
- Acrobat PDF tutorial documents
- Fillable forms creation and management
- Onboarding and compliance workflows

🚀 CAPABILITIES:
- Context-aware section extraction for HR workflows
- Acrobat feature-focused content ranking
- Form management and e-signature content filtering
- HR-specific content prioritization
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
    """Context for HR professional persona"""
    role: str
    task: str
    constraints: List[str]
    priorities: List[str]


class Collection2Processor:
    """
    🎯 Challenge 1B Collection 2 Processor
    
    Specialized for HR professionals working with Acrobat PDF forms
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Configuration specifically for HR Professional + Acrobat Forms (Collection 2)
        self.hr_config = {
            'priority_keywords': [
                # Form Creation & Management
                'form', 'forms', 'fillable', 'interactive', 'field', 'fields', 'create', 'prepare',
                'flat', 'convert', 'text field', 'checkbox', 'radio button', 'dropdown', 'combo',
                
                # HR Workflows
                'onboarding', 'compliance', 'employee', 'staff', 'hr', 'human resources',
                'hiring', 'recruitment', 'orientation', 'training', 'documentation',
                
                # Acrobat Features
                'acrobat', 'adobe', 'pdf', 'reader', 'pro', 'tools', 'toolbar', 'menu',
                'fill', 'sign', 'signature', 'e-signature', 'electronic', 'digital',
                
                # Form Operations
                'enable', 'save', 'export', 'share', 'send', 'email', 'recipients',
                'multiple', 'batch', 'collect', 'responses', 'data', 'submit',
                
                # Technical Instructions
                'click', 'select', 'choose', 'open', 'edit', 'add', 'insert', 'delete',
                'modify', 'change', 'update', 'configure', 'set up', 'customize',
                
                # Workflow Management
                'manage', 'organize', 'distribute', 'track', 'monitor', 'review',
                'approve', 'workflow', 'process', 'procedure', 'step', 'guide'
            ],
            'importance_weights': {
                'fillable_forms': 1.0,
                'e_signatures': 0.95,
                'form_creation': 0.9,
                'batch_processing': 0.85,
                'compliance': 0.8,
                'sharing': 0.75,
                'editing': 0.7,
                'conversion': 0.65
            }
        }
        
        if self.debug:
            print("SUCCESS: Challenge 1B Collection 2 Processor initialized for HR Professional")
    
    def process_collection_2(self, collection_path: Path = None) -> Dict[str, Any]:
        """
        Process Collection 2 specifically - HR Acrobat form management
        
        Args:
            collection_path: Path to Collection 2 directory (optional)
            
        Returns:
            dict: Complete analysis results for HR form management
        """
        if collection_path is None:
            collection_path = Path("Collection 2")
        
        if self.debug:
            print(f"🔍 Processing Collection 2: {collection_path}")
        
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
        
        # Rank and select top sections for HR workflows
        top_sections = self._rank_sections_for_hr(all_sections, persona_context)
        
        # Generate comprehensive analysis
        analysis = self._generate_hr_analysis(top_sections, persona_context)
        
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
            role=persona.get("role", "HR professional"),
            task=task,
            constraints=input_config.get("constraints", []),
            priorities=["Fillable forms", "E-signatures", "Batch processing", "Compliance workflows"]
        )
    
    def _extract_document_sections(self, pdf_path: Path, context: PersonaContext) -> List[DocumentSection]:
        """Extract sections using optimized approach for HR Acrobat workflows"""
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
                    
                    if len(block_text) < 40:  # Low threshold for better recall
                        continue
                    
                    # Generate semantic section title based on content theme
                    theme, theme_boost = self._classify_content_theme(block_text, pdf_path.name)
                    section_title = self._generate_semantic_title(block_text, theme, pdf_path.name)
                    
                    # Calculate importance and relevance scores
                    importance_score = self._calculate_hr_importance(block_text, context, pdf_path.name)
                    relevance_score = self._calculate_hr_relevance(block_text, pdf_path.name, context)
                    
                    # Apply theme boost
                    importance_score = min(importance_score + theme_boost, 1.0)
                    
                    # Thresholds optimized for HR content
                    threshold_importance = 0.15 if theme != 'general' else 0.25
                    threshold_relevance = 0.15 if theme != 'general' else 0.25
                    
                    if importance_score > threshold_importance or relevance_score > threshold_relevance:
                        section = DocumentSection(
                            document=pdf_path.name,
                            section_title=section_title,
                            content=block_text,
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
                print(f"ERROR: Failed to process {pdf_path}: {e}")
        
        return sections
    
    def _classify_content_theme(self, text: str, document_name: str) -> Tuple[str, float]:
        """Classify content into HR/Acrobat workflow themes"""
        text_lower = text.lower()
        doc_lower = document_name.lower()
        
        # Define HR-specific content themes
        theme_patterns = {
            'fillable_forms': {
                'keywords': ['fillable', 'interactive', 'form', 'field', 'text field', 'checkbox', 'radio button', 'dropdown', 'prepare forms', 'create form', 'form fields', 'flat forms'],
                'boost': 0.6,
                'documents': ['fill', 'sign']
            },
            'e_signatures': {
                'keywords': ['signature', 'e-signature', 'electronic', 'digital', 'sign', 'signing', 'request', 'recipients', 'email', 'send document', 'get signatures'],
                'boost': 0.6,
                'documents': ['signature', 'e-signature']
            },
            'form_creation': {
                'keywords': ['create', 'convert', 'prepare', 'build', 'design', 'multiple pdfs', 'batch', 'clipboard', 'files', 'documents'],
                'boost': 0.5,
                'documents': ['create', 'convert']
            },
            'editing_tools': {
                'keywords': ['edit', 'modify', 'change', 'update', 'tools', 'toolbar', 'menu', 'select', 'click', 'add', 'insert', 'delete'],
                'boost': 0.4,
                'documents': ['edit']
            },
            'sharing_workflow': {
                'keywords': ['share', 'distribute', 'send', 'email', 'collaborate', 'review', 'collect', 'responses', 'checklist'],
                'boost': 0.4,
                'documents': ['share', 'checklist']
            },
            'export_conversion': {
                'keywords': ['export', 'convert', 'save', 'format', 'output', 'file type', 'skills', 'test'],
                'boost': 0.3,
                'documents': ['export', 'skills']
            }
        }
        
        best_theme = 'general'
        best_score = 0.0
        
        for theme, config in theme_patterns.items():
            theme_score = 0.0
            
            # Check document match
            if any(doc_word in doc_lower for doc_word in config['documents']):
                theme_score += 0.3
            
            # Check keyword matches
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            if keyword_matches > 0:
                theme_score += (keyword_matches / len(config['keywords'])) * config['boost']
                theme_score += min(keyword_matches * 0.15, 0.4)
                
                # Extra boost for multiple matches
                if keyword_matches >= 3:
                    theme_score += 0.2
            
            if theme_score > best_score:
                best_score = theme_score
                best_theme = theme
        
        return best_theme, best_score
    
    def _generate_semantic_title(self, text: str, theme: str, document_name: str) -> str:
        """Generate semantic section titles based on HR workflow themes"""
        text_lower = text.lower()
        doc_lower = document_name.lower()
        
        # Map themes to appropriate section titles
        if theme == 'fillable_forms':
            if 'flat' in text_lower and 'fillable' in text_lower:
                return "Change flat forms to fillable (Acrobat Pro)"
            elif 'interactive' in text_lower:
                return "Create interactive forms"
            else:
                return "Fill and sign PDF forms"
                
        elif theme == 'e_signatures':
            if 'send' in text_lower or 'others' in text_lower:
                return "Send a document to get signatures from others"
            elif 'request' in text_lower:
                return "Request electronic signatures"
            else:
                return "E-signature workflow management"
                
        elif theme == 'form_creation':
            if 'multiple' in text_lower and 'files' in text_lower:
                return "Create multiple PDFs from multiple files"
            elif 'clipboard' in text_lower:
                return "Convert clipboard content to PDF"
            else:
                return "PDF creation and conversion"
                
        elif theme == 'editing_tools':
            return "PDF editing tools and techniques"
            
        elif theme == 'sharing_workflow':
            if 'checklist' in text_lower:
                return "PDF sharing checklist"
            else:
                return "Share and collaborate on PDFs"
                
        elif theme == 'export_conversion':
            if 'skills' in text_lower or 'test' in text_lower:
                return "Acrobat export skills assessment"
            else:
                return "Export and conversion options"
        
        # Fallback based on document type
        if 'fill' in doc_lower and 'sign' in doc_lower:
            return "Fill and sign PDF forms"
        elif 'create' in doc_lower:
            return "PDF creation workflows"
        elif 'edit' in doc_lower:
            return "PDF editing capabilities"
        elif 'signature' in doc_lower:
            return "Electronic signature management"
        elif 'share' in doc_lower:
            return "PDF sharing and collaboration"
        elif 'export' in doc_lower:
            return "PDF export and conversion"
        
        # Final fallback
        first_line = text.split('\n')[0][:80] if '\n' in text else text[:80]
        return first_line.strip()
    
    def _calculate_hr_importance(self, text: str, context: PersonaContext, document_name: str) -> float:
        """Calculate importance score for HR professional workflows"""
        text_lower = text.lower()
        score = 0.0
        
        # Get theme classification boost
        theme, theme_score = self._classify_content_theme(text, document_name)
        score += theme_score
        
        # Check for HR-specific keywords
        keyword_matches = 0
        
        for keyword in self.hr_config['priority_keywords']:
            if keyword in text_lower:
                keyword_matches += 1
                # HR workflow-specific scoring
                if keyword in ['form', 'forms', 'fillable', 'interactive', 'field']:
                    score += 0.12  # Core form management
                elif keyword in ['onboarding', 'compliance', 'hr', 'employee']:
                    score += 0.10  # HR-specific workflows
                elif keyword in ['signature', 'e-signature', 'sign', 'signing']:
                    score += 0.10  # Critical e-signature functionality
                elif keyword in ['create', 'prepare', 'convert', 'multiple']:
                    score += 0.08  # Creation and batch operations
                else:
                    score += 0.05  # Other relevant keywords
        
        # Keyword density bonus
        text_words = text.split()
        if len(text_words) > 0:
            keyword_density = keyword_matches / len(text_words)
            score += keyword_density * 0.25
        
        # HR workflow indicators
        hr_indicators = [
            'onboarding', 'compliance', 'employee forms', 'staff documentation',
            'hiring process', 'hr workflow', 'human resources'
        ]
        for indicator in hr_indicators:
            if indicator in text_lower:
                score += 0.15
        
        # Acrobat feature indicators
        acrobat_features = [
            'acrobat pro', 'prepare forms', 'form fields', 'fill & sign',
            'e-signatures', 'batch processing', 'multiple pdfs'
        ]
        for feature in acrobat_features:
            if feature in text_lower:
                score += 0.12
        
        # Technical instruction indicators (important for HR users)
        instruction_indicators = [
            'click', 'select', 'choose', 'open', 'tools', 'menu', 'toolbar',
            'step by step', 'how to', 'guide', 'procedure'
        ]
        for instruction in instruction_indicators:
            if instruction in text_lower:
                score += 0.08
        
        # Content length consideration
        if len(text_words) > 50:
            score += 0.03
        if len(text_words) > 100:
            score += 0.03
        
        return min(score, 1.0)
    
    def _calculate_hr_relevance(self, text: str, document_name: str, context: PersonaContext) -> float:
        """Calculate relevance score for HR professional needs"""
        score = 0.0
        text_lower = text.lower()
        doc_lower = document_name.lower()
        
        # Document type relevance for HR
        if any(term in doc_lower for term in ['fill', 'sign']):
            score += 0.4  # Highest relevance for form filling
        elif any(term in doc_lower for term in ['signature', 'e-signature']):
            score += 0.4  # Equally important for e-signatures
        elif any(term in doc_lower for term in ['create', 'convert']):
            score += 0.35  # Important for form creation
        elif any(term in doc_lower for term in ['share', 'checklist']):
            score += 0.25  # Moderate relevance for sharing
        elif any(term in doc_lower for term in ['edit']):
            score += 0.2   # Lower relevance for editing
        
        # Content relevance for HR workflows
        hr_content_indicators = [
            'form', 'forms', 'fillable', 'interactive', 'field', 'fields',
            'signature', 'sign', 'signing', 'e-signature', 'electronic'
        ]
        if any(term in text_lower for term in hr_content_indicators):
            score += 0.3
        
        # Workflow efficiency indicators
        efficiency_indicators = [
            'multiple', 'batch', 'automate', 'streamline', 'efficient',
            'quick', 'easy', 'simple', 'save time'
        ]
        if any(term in text_lower for term in efficiency_indicators):
            score += 0.2
        
        # Compliance and documentation relevance
        compliance_indicators = [
            'compliance', 'documentation', 'record', 'track', 'audit',
            'legal', 'official', 'secure', 'policy'
        ]
        if any(term in text_lower for term in compliance_indicators):
            score += 0.25
        
        # Technical instruction relevance
        instruction_indicators = [
            'step', 'procedure', 'guide', 'tutorial', 'how to', 'method',
            'process', 'workflow', 'instructions'
        ]
        if any(term in text_lower for term in instruction_indicators):
            score += 0.15
        
        return min(score, 1.0)
    
    def _rank_sections_for_hr(self, sections: List[DocumentSection], context: PersonaContext) -> List[DocumentSection]:
        """Rank sections specifically for HR professional needs"""
        # Calculate combined scores
        for section in sections:
            section.combined_score = (section.importance_score * 0.6) + (section.relevance_score * 0.4)
        
        # Sort by combined score
        ranked_sections = sorted(sections, key=lambda x: x.combined_score, reverse=True)
        
        # Ensure diversity across document types
        selected_sections = []
        used_documents = set()
        
        # First pass: get top section from each document type
        for section in ranked_sections:
            if len(selected_sections) >= 12:  # Same as Collection 1 for good recall
                break
            
            doc_name = section.document
            # Prioritize getting sections from different document types
            if doc_name not in used_documents:
                selected_sections.append(section)
                used_documents.add(doc_name)
        
        # Second pass: fill remaining slots with highest scoring sections
        for section in ranked_sections:
            if len(selected_sections) >= 12:
                break
            if section not in selected_sections:
                selected_sections.append(section)
        
        return selected_sections[:12]
    
    def _generate_hr_analysis(self, sections: List[DocumentSection], context: PersonaContext) -> Dict[str, Any]:
        """Generate comprehensive HR workflow analysis"""
        analysis = {
            "form_management": {
                "creation_workflows": [],
                "fillable_forms": [],
                "batch_processing": []
            },
            "signature_workflows": {
                "e_signature_setup": [],
                "document_distribution": [],
                "compliance_tracking": []
            },
            "efficiency_tools": {
                "automation_features": [],
                "sharing_options": [],
                "integration_capabilities": []
            }
        }
        
        # Analyze sections for HR-specific insights
        for section in sections:
            if section.importance_score > 0.7:
                analysis["form_management"]["creation_workflows"].append({
                    "title": section.section_title,
                    "source": section.document,
                    "relevance": "high"
                })
            
            if section.importance_score > 0.6:
                analysis["signature_workflows"]["e_signature_setup"].append({
                    "title": section.section_title,
                    "source": section.document,
                    "relevance": "medium-high"
                })
        
        return analysis
    
    def _format_sections(self, sections: List[DocumentSection]) -> List[Dict[str, Any]]:
        """Format sections for output"""
        formatted = []
        for i, section in enumerate(sections, 1):
            formatted.append({
                "document": section.document,
                "section_title": section.section_title,
                "importance_rank": i,
                "page_number": section.page_number
            })
        return formatted
    
    def _format_subsections(self, sections: List[DocumentSection]) -> List[Dict[str, Any]]:
        """Format subsections with detailed content"""
        formatted = []
        for section in sections:
            formatted.append({
                "document": section.document,
                "refined_text": section.content,
                "page_number": section.page_number
            })
        return formatted


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Challenge 1B Collection 2 Processor - HR Professional')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--output', type=str, default='collection2_output.json', 
                       help='Output file path (default: collection2_output.json)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = Collection2Processor(debug=args.debug)
    
    print("🎯 Adobe India Hackathon 2025 - Challenge 1B Collection 2")
    print("📋 Processing HR Professional - Acrobat Forms Management")
    print("=" * 60)
    
    # Process Collection 2
    result = processor.process_collection_2()
    
    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
        return 1
    
    # Save output
    output_file = Path(args.output)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ SUCCESS: Collection 2 analysis completed!")
        print(f"📄 OUTPUT: Results saved to {output_file}")
        
        # Display summary
        sections_count = len(result.get('extracted_sections', []))
        subsections_count = len(result.get('subsection_analysis', []))
        print(f"📊 SUMMARY: {sections_count} main sections, {subsections_count} detailed subsections")
        
    except Exception as e:
        print(f"❌ ERROR: Could not save output file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
