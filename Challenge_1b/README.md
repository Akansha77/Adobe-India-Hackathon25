# Challenge 1b: Multi-Collection PDF Analysis

## Overview
Advanced PDF analysis solution that processes multiple document collections and extracts relevant content based on specific personas and use cases.

## Project Structure
```
Challenge_1b/
├── Collection 1/                    # Travel Planning
│   ├── PDFs/                       # South of France guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── Collection 2/                    # Adobe Acrobat Learning
│   ├── PDFs/                       # Acrobat tutorials
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── Collection 3/                    # Recipe Collection
│   ├── PDFs/                       # Cooking guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
└── README.md
```

## Collections

### Collection 1: Travel Planning
- **Challenge ID**: round_1b_002
- **Persona**: Travel Planner
- **Task**: Plan a 4-day trip for 10 college friends to South of France
- **Documents**: 7 travel guides

### Collection 2: Adobe Acrobat Learning
- **Challenge ID**: round_1b_003
- **Persona**: HR Professional
- **Task**: Create and manage fillable forms for onboarding and compliance
- **Documents**: 15 Acrobat guides

### Collection 3: Recipe Collection
- **Challenge ID**: round_1b_001
- **Persona**: Food Contractor
- **Task**: Prepare vegetarian buffet-style dinner menu for corporate gathering
- **Documents**: 9 cooking guides

## Input/Output Format

### Input JSON Structure
```json
{
  "challenge_info": {
    "challenge_id": "round_1b_XXX",
    "test_case_name": "specific_test_case"
  },
  "documents": [{"filename": "doc.pdf", "title": "Title"}],
  "persona": {"role": "User Persona"},
  "job_to_be_done": {"task": "Use case description"}
}
```

### Output JSON Structure
```json
{
  "metadata": {
    "input_documents": ["list"],
    "persona": "User Persona",
    "job_to_be_done": "Task description"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Content",
      "page_number": 1
    }
  ]
}
```

## Key Features
- Persona-based content analysis
- Importance ranking of extracted sections
- Multi-collection document processing
- Structured JSON output with metadata

---

## 🚀 IMPLEMENTATION - Challenge 1B Solution

### Features

✅ **Persona-based Analysis**: Content extraction tailored to specific user roles and tasks  
✅ **Importance Ranking**: Sections ranked by relevance to the persona's objectives  
✅ **Multi-document Processing**: Handles multiple PDFs simultaneously  
✅ **Structured Output**: JSON format matching challenge specifications  
✅ **Context-aware Extraction**: Considers document type and task requirements  

### Quick Start

#### Process Single Collection
```bash
# Process Collection 1 (Travel Planning)
python challenge1b_processor.py --collection "Collection 1" --debug

# Process Collection 2 (HR Professional)
python challenge1b_processor.py --collection "Collection 2" 

# Process Collection 3 (Food Contractor)
python challenge1b_processor.py --collection "Collection 3"
```

#### Process All Collections
```bash
# Process all collections at once
python run_collections.py --all

# Process specific collection with batch runner
python run_collections.py --collection 1 --debug
```

### Algorithm Details

#### 1. Persona-based Scoring
Each persona has specific priority keywords and importance weights:

- **Travel Planner**: accommodation, activities, budget, group planning
- **HR Professional**: forms, compliance, automation, workflows  
- **Food Contractor**: vegetarian, buffet, menu planning, ingredients

#### 2. Section Extraction
- Identifies section headings using font analysis and content patterns
- Extracts content blocks with semantic structure
- Calculates importance and relevance scores per section

#### 3. Importance Ranking
Sections ranked by combined score:
- **Importance Score (60%)**: Keyword matching + task relevance
- **Relevance Score (40%)**: Document type + content quality

### Performance

- **Processing Speed**: ~0.2-0.3 seconds per collection
- **Section Extraction**: 200-300 sections identified per collection
- **Top Sections**: 5 most important sections ranked
- **Analysis Depth**: 10 detailed subsection analyses

### Implementation Files

- `challenge1b_processor.py` - Main processor with persona-based analysis
- `run_collections.py` - Batch runner for multiple collections
- Generated `challenge1b_output.json` files in each collection directory

---

**Note**: This README provides a brief overview of the Challenge 1b solution structure based on available sample data. 