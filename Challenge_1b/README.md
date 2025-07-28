# Challenge 1b: Multi-Collection PDF Intelligence

## 🎯 Objective
Extract relevant content from PDFs based on user personas and specific tasks across multiple domains.

## 📊 Performance Results
- **Average Precision:** 50.0%
- **Average Recall:** 40.0%
- **Average F1-Score:** 44.4%
- **Rank Accuracy:** 100.0%

## 🔍 Collections
| Collection | Persona | Task | F1-Score |
|------------|---------|------|----------|
| Collection 1 | Travel Planner | Plan 4-day trip | 88.9% |
| Collection 2 | HR Professional | Create training content | 44.4% |
| Collection 3 | Food Contractor | Design buffet menu | 0.0% |

## 🛠️ Usage
```bash
# Run all collections
python run_collections.py

# Evaluate results
python evaluate_challenge1b.py
```

## 📁 Structure
```
Challenge_1b/
├── run_collections.py          # Main runner
├── evaluate_challenge1b.py     # Evaluation script
├── Collection 1/               # Travel planning
├── Collection 2/               # HR training
└── Collection 3/               # Food service
```
