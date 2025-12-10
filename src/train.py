"""
Model Training Module
=====================
Train and compare multiple models for flight delay prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, will skip XGBoost model")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of models to train.
    
    Returns:
        Dictionary of model name -> model instance
    """
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=1,  # Will be updated based on class imbalance
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
    
    return models


def split_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'delayed',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        test_size: Proportion for test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    print(f"Data split:")
    print(f"  Train: {len(X_train):,} samples ({y_train.mean()*100:.1f}% delayed)")
    print(f"  Test:  {len(X_test):,} samples ({y_test.mean()*100:.1f}% delayed)")
    
    return X_train, X_test, y_train, y_test


def train_model(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a single model with cross-validation.
    
    Args:
        model: Model instance
        model_name: Name for logging
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
    
    Returns:
        Trained model and CV scores dictionary
    """
    print(f"\nTraining {model_name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    
    print(f"  CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Train on full training data
    model.fit(X_train, y_train)
    
    scores = {
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std()
    }
    
    return model, scores


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Dict[str, Tuple[Any, Dict[str, float]]]:
    """
    Train all models and return results.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
    
    Returns:
        Dictionary of model name -> (trained model, scores)
    """
    print("=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)
    
    models = get_models()
    results = {}
    
    for name, model in models.items():
        trained_model, scores = train_model(model, name, X_train, y_train, cv)
        results[name] = (trained_model, scores)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    
    return results


def save_model(model: Any, model_name: str) -> Path:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name for the file
    
    Returns:
        Path to saved model
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean name for filename
    filename = model_name.lower().replace(' ', '_') + '.joblib'
    filepath = MODELS_DIR / filename
    
    joblib.dump(model, filepath)
    print(f"Saved model to {filepath}")
    
    return filepath


def save_feature_columns(feature_cols: List[str]) -> Path:
    """
    Save feature column names for use in prediction.
    
    Args:
        feature_cols: List of feature column names
    
    Returns:
        Path to saved file
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / "feature_columns.joblib"
    
    joblib.dump(feature_cols, filepath)
    print(f"Saved feature columns to {filepath}")
    
    return filepath


def load_feature_columns() -> List[str]:
    """Load saved feature columns."""
    filepath = MODELS_DIR / "feature_columns.joblib"
    if filepath.exists():
        return joblib.load(filepath)
    return []


def load_model(model_name: str) -> Any:
    """Load a saved model."""
    filename = model_name.lower().replace(' ', '_') + '.joblib'
    filepath = MODELS_DIR / filename
    return joblib.load(filepath)


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """
    Extract feature importances from a trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model (for handling different types)
    
    Returns:
        DataFrame with feature names and importances, sorted by importance
    """
    if model_name == 'Logistic Regression':
        # For pipeline, get classifier step
        classifier = model.named_steps['classifier']
        importances = np.abs(classifier.coef_[0])
    elif model_name in ['Random Forest', 'XGBoost']:
        importances = model.feature_importances_
    else:
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


if __name__ == "__main__":
    # Quick test
    print("Train module loaded successfully")
    print(f"Available models: {list(get_models().keys())}")
