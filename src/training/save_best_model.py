"""
Save the best model + feature engineer state for making predictions
"""
import os
import sys
from src.training.enhanced_features import EnhancedFeatureEngineer
from src.core.sumo_predictor import ModelConfig, SumoDataLoader
from src.utils.gpu_optimizer import GPUOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import joblib
from datetime import datetime
import numpy as np

def train_and_save_production_model(start_basho=None, end_basho=None, latest_basho=None, latest_day=None, verbose=True, use_gpu=False):
    """
    Train and save the production model.

    Args:
        start_basho: Starting basho ID (None = all data)
        end_basho: Ending basho ID (None = all data)
        latest_basho: Latest basho ID in database (for metadata)
        latest_day: Latest day in database (for metadata)
        verbose: Print progress messages
        use_gpu: Enable GPU acceleration (default False - CPU is faster for this dataset)

    Returns:
        dict with training results or None on error
    """
    if verbose:
        print("="*80)
        print("TRAINING AND SAVING BEST MODEL FOR PREDICTIONS")
        print("="*80)

    # GPU optimization (disabled by default - CPU is faster for this dataset size)
    # Benchmark showed CPU is 3.1x faster for ~82K samples
    # See GPU_BENCHMARK_RESULTS.md for details
    if use_gpu:
        if verbose:
            print("\nDetecting GPU hardware and optimizations...")
        gpu_optimizer = GPUOptimizer()
        if verbose:
            gpu_optimizer.print_summary()
    else:
        gpu_optimizer = None
        if verbose:
            print("\nUsing CPU training (faster for this dataset size)")

    # Load historical data
    if verbose:
        print("\nLoading historical data...")
    config = ModelConfig(elo_k_factor=32, recent_bouts_window=15)
    loader = SumoDataLoader()
    bouts_df = loader.load_raw_bouts()

    # Filter by basho range if specified
    if start_basho is not None:
        bouts_df = bouts_df[bouts_df['basho_id'] >= start_basho]
    if end_basho is not None:
        bouts_df = bouts_df[bouts_df['basho_id'] <= end_basho]

    if len(bouts_df) == 0:
        if verbose:
            print("ERROR: No bouts found in specified range")
        return None

    # Build features (this processes all historical bouts and updates Elo/stats)
    if verbose:
        print("Building features from historical data...")
    engineer = EnhancedFeatureEngineer(config)
    X, y = engineer.build_dataset(bouts_df)

    if verbose:
        print(f"\nTrained on {len(bouts_df)} bouts")
        print(f"Feature engineer now has:")
        print(f"  - Elo ratings for {len(engineer.elo_system.ratings)} wrestlers")
        print(f"  - Historical stats for {len(engineer.rikishi_stats)} wrestlers")

    # Perform cross-validation to get reliable accuracy estimate
    if verbose:
        print("\nPerforming 5-fold cross-validation...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_weights = {'rf': 0.45, 'lgb': 0.45, 'xgb': 0.10}

    # Get optimized parameters
    if use_gpu and gpu_optimizer:
        recommendations = gpu_optimizer.get_training_recommendations()
        n_estimators = recommendations.get('recommended_n_estimators', 400)
    else:
        n_estimators = 400  # CPU-optimized default

    # Create models for CV
    rf_model_cv = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # LightGBM configuration
    lgb_base_params = {
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': n_estimators,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    if use_gpu and gpu_optimizer:
        lgb_params = gpu_optimizer.get_lightgbm_params(lgb_base_params)
    else:
        lgb_params = {**lgb_base_params, 'n_jobs': -1}  # CPU multi-threading
    lgb_model_cv = lgb.LGBMClassifier(**lgb_params)

    # XGBoost configuration
    xgb_base_params = {
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': n_estimators,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    if use_gpu and gpu_optimizer:
        xgb_params = gpu_optimizer.get_xgboost_params(xgb_base_params)
    else:
        xgb_params = {**xgb_base_params, 'n_jobs': -1}  # CPU multi-threading
    xgb_model_cv = xgb.XGBClassifier(**xgb_params)

    if verbose:
        print("  [1/3] Random Forest CV...")
    rf_scores = cross_val_score(rf_model_cv, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    if verbose:
        print("  [2/3] LightGBM CV...")
    lgb_scores = cross_val_score(lgb_model_cv, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    if verbose:
        print("  [3/3] XGBoost CV...")
    xgb_scores = cross_val_score(xgb_model_cv, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    # Calculate weighted ensemble CV accuracy
    ensemble_cv_scores = (
        ensemble_weights['rf'] * rf_scores +
        ensemble_weights['lgb'] * lgb_scores +
        ensemble_weights['xgb'] * xgb_scores
    )
    accuracy = ensemble_cv_scores.mean()

    if verbose:
        print(f"\n5-Fold Cross-Validation Results:")
        print(f"  Random Forest:  {rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})")
        print(f"  LightGBM:       {lgb_scores.mean():.4f} (+/- {lgb_scores.std() * 2:.4f})")
        print(f"  XGBoost:        {xgb_scores.mean():.4f} (+/- {xgb_scores.std() * 2:.4f})")
        print(f"  Ensemble:       {accuracy:.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")

    # Train final models on ALL data for production
    if verbose:
        print("\nTraining final models on ALL data for production...")

    if verbose:
        print("  [1/3] Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)

    if verbose:
        print("  [2/3] LightGBM...")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X, y)

    if verbose:
        print("  [3/3] XGBoost...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X, y)

    # Save everything needed for predictions
    if verbose:
        print("\nSaving model package...")
    model_package = {
        'config': config,
        'feature_engineer': engineer,  # Contains Elo ratings, stats, etc.
        'models': {
            'random_forest': rf_model,
            'lightgbm': lgb_model,
            'xgboost': xgb_model
        },
        'ensemble_weights': ensemble_weights,
        'feature_names': list(X.columns),
        'training_date': datetime.now().isoformat(),
        'num_training_bouts': len(bouts_df),
        'accuracy': accuracy,
        'last_trained_basho_id': latest_basho if latest_basho else end_basho,
        'last_trained_day': latest_day
    }

    # Get project root and save to models directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, 'models')
    filename = os.path.join(models_dir, 'sumo_predictor_production.joblib')

    joblib.dump(model_package, filename)

    if verbose:
        print(f"\n✅ Model saved to: {filename}")
        print(f"\nThis package contains:")
        print(f"  - Trained ensemble models (RF + LGB + XGB)")
        print(f"  - Feature engineer with Elo ratings and historical stats")
        print(f"  - Configuration and metadata")
        print(f"\n5-Fold CV Ensemble Accuracy: {accuracy*100:.2f}% (+/- {ensemble_cv_scores.std() * 2 * 100:.2f}%)")
        print("="*80)

    return {
        'num_training_bouts': len(bouts_df),
        'accuracy': accuracy,
        'model_path': filename
    }

if __name__ == "__main__":
    train_and_save_production_model()
