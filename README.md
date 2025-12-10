# Flight Delay Prediction

A machine learning pipeline to predict flight delays using classification models. This project demonstrates data cleaning, feature engineering, model training, and explainability.

## Project Overview

This project uses the [Kaggle Flight Delays Dataset](https://www.kaggle.com/datasets/usdot/flight-delays) to build binary classifiers that predict whether a flight will be delayed (arrival delay > 15 minutes).

### Models Trained
- **Logistic Regression** - Baseline interpretable model
- **Random Forest** - Ensemble method with feature importances
- **XGBoost** - Gradient boosting for best performance

### Features Engineered
- **Time features**: Hour of day, time of day category, weekend indicator
- **Congestion features**: Flights per hour at origin/destination airports
- **Airline features**: Historical delay rate per airline
- **Route features**: Route popularity

## Project Structure

```
flight/
├── data/
│   ├── raw/                    # Original dataset (flights.csv, airlines.csv, airports.csv)
│   └── processed/              # Cleaned, sampled data
├── notebooks/                  # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocessing.py        # Cleaning and target creation
│   ├── features.py             # Feature engineering
│   ├── train.py                # Model training
│   └── evaluate.py             # Evaluation and visualization
├── models/                     # Saved trained models (.joblib)
├── outputs/
│   ├── figures/                # Confusion matrices, ROC curves, feature importances
│   └── reports/                # Model comparison CSV
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── main.py                     # Main pipeline runner
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays) and place files in `data/raw/`:
- `flights.csv`
- `airlines.csv`
- `airports.csv`

### 3. Run Pipeline

```bash
# Run with defaults (50K samples)
python main.py

# Quick test (5K samples)
python main.py --quick-test

# Custom options
python main.py --sample-size 100000 --delay-threshold 20
```

## Results

After running the pipeline, check:

| Output | Location |
|--------|----------|
| Trained models | `models/*.joblib` |
| Confusion matrices | `outputs/figures/confusion_matrix_*.png` |
| ROC curves | `outputs/figures/roc_curves.png` |
| Feature importance | `outputs/figures/feature_importance_*.png` |
| Model comparison | `outputs/reports/model_comparison.csv` |

### Sample Performance (50K samples)

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~86% | ~0.68 | ~0.88 |
| Random Forest | ~89% | ~0.73 | ~0.89 |
| XGBoost | ~91% | ~0.75 | ~0.89 |

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sample-size` | 50000 | Number of samples to use |
| `--quick-test` | False | Run with 5000 samples, 3-fold CV |
| `--delay-threshold` | 15 | Minutes threshold for delay |
| `--test-size` | 0.2 | Test set proportion |
| `--cv-folds` | 5 | Cross-validation folds |
| `--skip-viz` | False | Skip visualization generation |

## Dependencies

- pandas, numpy - Data processing
- scikit-learn - ML algorithms
- xgboost - Gradient boosting
- matplotlib, seaborn - Visualization
- joblib - Model persistence

## License

MIT License - Feel free to use for learning and portfolio purposes.
