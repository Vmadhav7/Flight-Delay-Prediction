"""
Flight Delay Prediction - Main Pipeline
========================================

This script runs the complete ML pipeline:
1. Load and sample data
2. Preprocess and clean
3. Engineer features
4. Train models
5. Evaluate and visualize

Usage:
    python main.py                    # Run with defaults
    python main.py --sample-size 100000  # Custom sample size
    python main.py --quick-test       # Quick test with 5000 samples
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import load_raw_flights, sample_data, save_processed
from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline
from src.train import split_data, train_all_models, save_model, get_feature_importance, save_feature_columns
from src.evaluate import (
    evaluate_all_models,
    save_model_comparison,
    generate_all_visualizations
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Flight Delay Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50000,
        help='Number of samples to use (default: 50000)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with 5000 samples'
    )
    
    parser.add_argument(
        '--delay-threshold',
        type=int,
        default=15,
        help='Minutes threshold for delay classification (default: 15)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Cross-validation folds (default: 5)'
    )
    
    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    return parser.parse_args()


def main():
    """Run the complete pipeline."""
    args = parse_args()
    
    # Override sample size for quick test
    if args.quick_test:
        args.sample_size = 5000
        args.cv_folds = 3
        print("[QUICK TEST] Running in QUICK TEST mode (5000 samples, 3-fold CV)")
    
    print("\n" + "=" * 60)
    print("   FLIGHT DELAY PREDICTION PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Sample size:      {args.sample_size:,}")
    print(f"  Delay threshold:  {args.delay_threshold} minutes")
    print(f"  Test size:        {args.test_size*100:.0f}%")
    print(f"  CV folds:         {args.cv_folds}")
    print()
    
    # =========================================
    # STEP 1: Load Data
    # =========================================
    print("\n[STEP 1] Loading Data")
    print("-" * 40)
    
    # Load a larger chunk to sample from
    load_size = min(args.sample_size * 3, 500000)  # Load 3x samples for stratified sampling
    df = load_raw_flights(nrows=load_size)
    
    # =========================================
    # STEP 2: Preprocess
    # =========================================
    print("\n[STEP 2] Preprocessing")
    print("-" * 40)
    
    df = preprocess_pipeline(df, delay_threshold=args.delay_threshold)
    
    # =========================================
    # STEP 3: Sample Data
    # =========================================
    print("\n[STEP 3] Sampling Data")
    print("-" * 40)
    
    df = sample_data(df, n_samples=args.sample_size)
    
    # =========================================
    # STEP 4: Feature Engineering
    # =========================================
    print("\n[STEP 4] Feature Engineering")
    print("-" * 40)
    
    df, feature_cols = feature_engineering_pipeline(df)
    
    # =========================================
    # STEP 5: Train/Test Split
    # =========================================
    print("\n[STEP 5] Train/Test Split")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = split_data(
        df, feature_cols,
        test_size=args.test_size
    )
    
    # =========================================
    # STEP 6: Train Models
    # =========================================
    print("\n[STEP 6] Training Models")
    print("-" * 40)
    
    trained_models = train_all_models(X_train, y_train, cv=args.cv_folds)
    
    # Save models
    print("\n[SAVING] Saving models...")
    for name, (model, _) in trained_models.items():
        save_model(model, name)
    
    # Save feature columns for prediction
    save_feature_columns(feature_cols)
    
    # =========================================
    # STEP 7: Evaluate
    # =========================================
    print("\n[STEP 7] Evaluation")
    print("-" * 40)
    
    results_df = evaluate_all_models(trained_models, X_test, y_test)
    save_model_comparison(results_df)
    
    # Find best model by F1 score
    best_model_name = results_df['f1'].idxmax()
    print(f"\n[BEST] Best Model: {best_model_name} (F1 = {results_df.loc[best_model_name, 'f1']:.4f})")
    
    # =========================================
    # STEP 8: Visualizations
    # =========================================
    if not args.skip_viz:
        print("\n[STEP 8] Generating Visualizations")
        print("-" * 40)
        
        generate_all_visualizations(
            trained_models, X_test, y_test,
            feature_cols, best_model_name
        )
    
    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"""
Outputs generated:
   - models/              : Saved model files (.joblib)
   - outputs/figures/     : Confusion matrices, ROC curves, feature importance
   - outputs/reports/     : Model comparison CSV

Best performing model: {best_model_name}
   - F1 Score: {results_df.loc[best_model_name, 'f1']:.4f}
   - Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}
   - ROC-AUC:  {results_df.loc[best_model_name, 'roc_auc']:.4f}
""")
    
    return results_df, trained_models


if __name__ == "__main__":
    results, models = main()
