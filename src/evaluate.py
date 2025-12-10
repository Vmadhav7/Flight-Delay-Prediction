"""
Evaluation Module
=================
Evaluate models and create visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, float]:
    """
    Evaluate a single model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        model_name: Name for logging
    
    Returns:
        Dictionary of metric name -> value
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    
    return metrics


def evaluate_all_models(
    trained_models: Dict[str, Tuple[Any, Dict]],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluate all trained models and return comparison DataFrame.
    
    Args:
        trained_models: Dictionary of model name -> (trained model, CV scores)
        X_test: Test features
        y_test: True labels
    
    Returns:
        DataFrame comparing all models
    """
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    results = []
    
    for name, (model, cv_scores) in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics['model'] = name
        metrics['cv_f1_mean'] = cv_scores['cv_f1_mean']
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('model')
    
    # Reorder columns
    col_order = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_f1_mean']
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order]
    
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(results_df.round(4).to_string())
    
    return results_df


def plot_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix for a model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        model_name: Name for title and filename
        save: Whether to save the figure
    
    Returns:
        Matplotlib figure
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Delayed', 'Delayed'],
        yticklabels=['Not Delayed', 'Delayed'],
        ax=ax
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / filename}")
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    top_n: int = 15,
    save: bool = True
) -> plt.Figure:
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        model_name: Name for title and filename
        top_n: Number of top features to show
        save: Whether to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Get top N features
    plot_df = importance_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bar chart
    colors = sns.color_palette("viridis", len(plot_df))
    bars = ax.barh(
        plot_df['feature'],
        plot_df['importance'],
        color=colors
    )
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Highest importance at top
    
    plt.tight_layout()
    
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / filename}")
    
    return fig


def plot_roc_curves(
    trained_models: Dict[str, Tuple[Any, Dict]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True
) -> plt.Figure:
    """
    Plot ROC curves for all models.
    
    Args:
        trained_models: Dictionary of model name -> (trained model, scores)
        X_test: Test features
        y_test: True labels
        save: Whether to save the figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, (model, _) in trained_models.items():
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'roc_curves.png'}")
    
    return fig


def save_model_comparison(results_df: pd.DataFrame) -> Path:
    """
    Save model comparison results to CSV.
    
    Args:
        results_df: DataFrame with model comparison metrics
    
    Returns:
        Path to saved file
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "model_comparison.csv"
    
    results_df.to_csv(output_path)
    print(f"Saved: {output_path}")
    
    return output_path


def generate_all_visualizations(
    trained_models: Dict[str, Tuple[Any, Dict]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    best_model_name: str = 'Random Forest'
) -> None:
    """
    Generate all evaluation visualizations.
    
    Args:
        trained_models: Dictionary of trained models
        X_test: Test features
        y_test: True labels
        feature_names: List of feature names
        best_model_name: Name of best model for detailed plots
    """
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    
    from train import get_feature_importance
    
    # 1. Confusion matrices for all models
    for name, (model, _) in trained_models.items():
        plot_confusion_matrix(model, X_test, y_test, name)
    
    # 2. ROC curves
    plot_roc_curves(trained_models, X_test, y_test)
    
    # 3. Feature importance for best model
    if best_model_name in trained_models:
        model, _ = trained_models[best_model_name]
        importance_df = get_feature_importance(model, feature_names, best_model_name)
        if not importance_df.empty:
            plot_feature_importance(importance_df, best_model_name)
    
    print("\n" + "=" * 50)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 50)
    
    plt.close('all')


if __name__ == "__main__":
    print("Evaluate module loaded successfully")
