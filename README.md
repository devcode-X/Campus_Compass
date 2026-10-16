# 🧭 Campus Compass

> **Never get lost again** — Your intelligent guide to navigating campus resources, facilities, and information — all in one place.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

---

## 📖 Overview

**Campus Compass** is a smart campus navigation and resource discovery tool designed to help students quickly find the information they need — from locating campus facilities and departments to surfacing relevant academic resources. Built as an interactive Jupyter Notebook, it provides a low-friction, data-driven way to explore everything a campus has to offer, without digging through endless portals and PDFs.

---

## ✨ Key Features

- 🗺️ **Facility finder** — Search and locate campus buildings, labs, libraries, cafeterias, and offices by name or category
- 📚 **Resource discovery** — Surface academic resources, support services, and departmental contacts in seconds
- 🔍 **Keyword search** — Natural-language-style query interface to find campus information intuitively
- 📊 **Data-driven** — Backed by structured campus datasets (CSV/JSON) for fast, accurate lookups
- 📓 **Interactive notebooks** — Jupyter-based UI allows exploratory, cell-by-cell interaction
- 🔧 **Easily extensible** — Add new facilities or resource categories by updating the data files

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Interface | Jupyter Notebook |
| Data Handling | pandas |
| Visualization | matplotlib / ipywidgets |
| Data Format | CSV / JSON |

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+
- **Jupyter Notebook** or **JupyterLab**

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/devcode-X/Campus_Compass.git
cd Campus_Compass

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Launch the Notebook

```bash
jupyter notebook Campus_Compass.ipynb
```

Then run the cells interactively — enter a building name, resource type, or keyword to get instant results.

---

## 📁 Folder Structure

```
Campus_Compass/
├── Campus_Compass.ipynb      # Main interactive notebook
├── data/
│   ├── facilities.csv        # Campus buildings & locations
│   └── resources.json        # Academic & support resources
├── utils/
│   └── search.py             # Search and filtering logic
├── assets/
│   └── campus_map.png        # Optional visual campus map
├── requirements.txt
└── README.md
```

---

## 🧪 Example Usage

```python
# Find all libraries on campus
search_facilities(query="library", category="academic")

# Locate the computer science department
get_resource(name="Computer Science Department")
```

**Output:**
```
📍 CS Department — Block C, Room 204
   Contact: cs-dept@university.edu | Hours: Mon–Fri 9am–5pm
```

---

## 👤 Author

**Devansh Awasthi** · [@devcode-X](https://github.com/devcode-X)
