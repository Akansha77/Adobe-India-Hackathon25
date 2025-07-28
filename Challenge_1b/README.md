# 📄 Adobe Hackathon 2025 – Challenge 1B: Multi-Collection PDF Intelligence

A smart PDF processing system that extracts **important content** based on **user roles (personas)** across multiple domains like travel planning, HR training, and food service.

---

## 🎯 Objective

Automatically process different sets of PDFs and extract the most **relevant sections** based on the persona’s goals.

### 🔍 Supported Personas:
| Collection | Persona           | Use Case |
|-----------|-------------------|----------|
| Collection 1 | Travel Planner     | Plan a 4-day trip for college students |
| Collection 2 | HR Professional    | Create Adobe Acrobat training content |
| Collection 3 | Food Contractor    | Design a vegetarian buffet menu |

---

## 📁 Folder Structure

Challenge_1b/
├── run_collections.py # Run all collections together
├── Collection 1/ # Travel planning PDFs
├── Collection 2/ # HR training PDFs
├── Collection 3/ # Recipe PDFs


Each collection contains:
- PDFs to process
- `collectionX_processor.py` → collection-specific processor
- `challenge1b_input.json` → persona & task info
- `collectionX_output.json` → output with extracted data

---

## 🚀 How to Run

### ✅ Run All Collections at Once
```bash
python run_collections.py

---

##▶️ Run Individual Collection
cd "Collection 1"
python collection1_processor.py


