# Predicting NCAA Tournament Success from Team Efficiency Metrics

## Overview
This project predicts NCAA Tournament outcomes using advanced college basketball efficiency metrics.  
It uses datasets containing team-level statistics such as adjusted offensive and defensive efficiency (**ADJOE**, **ADJDE**), effective field goal percentage (**EFG**), turnover rate (**TOR**), and tempo (**ADJ_T**).  
The main output includes predictive models that estimate tournament participation, seed assignment, and how far a team advances, along with visualizations and feature importance analyses.  
These insights aim to identify which metrics most strongly influence tournament success.

## Description
College basketball statistics have become increasingly advanced, with metrics like ADJOE, ADJDE, EFG, TOR, and ADJ_T offering a deeper look into team performance.  
While these numbers describe team strength, predicting tournament outcomes remains challenging.  
This project applies machine learning models — including logistic regression, random forests, and XGBoost — to predict NCAA outcomes from regular-season efficiency data.  

## Directory Structure
- `data/` – raw and processed datasets  
- `notebooks/` – Jupyter notebooks for exploration and visualization  
- `src/` – Python scripts for preprocessing, modeling, and evaluation  
- `results/` – generated figures and tables from analyses  
- `requirements.txt` – Python dependencies  

## Data
All datasets are stored in the `data/` folder:  
- `data/raw/` – raw CSV files downloaded from [Kaggle NCAA datasets](https://www.kaggle.com/datasets) (or other source)  
- `data/processed/` – cleaned and feature-engineered datasets used for modeling  
- File formats: CSV, UTF-8 encoded  
- Expected location: maintain folder structure as above  
- Size: small to moderate (~MB scale), no special access required  

### Download Instructions
1. Clone the repository:  
   ```bash
   git clone https://github.com/lambjos3/cmse492_project.git
   cd cmse492_project
2. Ensure the data/raw/ folder contains the raw CSVs (either download manually or place provided files here).
### Install uv if not already installed
pip install uv

### Initialize environment and install dependencies
uv add -r requirements.txt
uv lock

### Activate environment (example for venv)
### source .venv/bin/activate
required packages are: numpy, pandas, matplotlib, seaborn, and scikit-learn

## Reproducing a result
### Ensure you are in project root
cd cmse492_project

### Run preprocessing and model scripts
python src/preprocess.py
python src/train_models.py

# Generate figures and tables
python src/evaluate.py
