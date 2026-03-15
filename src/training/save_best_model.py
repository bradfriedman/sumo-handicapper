"""
Save the best model + feature engineer state for making predictions
"""
import os
import numpy as np
import pandas as pd
from typing import cast
from src.training.enhanced_features import EnhancedFeatureEngineer
from src.core.sumo_predictor import ModelConfig, SumoDataLoader
from src.utils.gpu_optimizer import GPUOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import joblib
from datetime import datetime


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def train_and_save_production_model(
    start_basho=None,
    end_basho=None,
    latest_basho=None,
    latest_day=None,
    verbose=True,
    use_gpu=False,
):
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
        print("=" * 80)
        print("TRAINING AND SAVING BEST MODEL FOR PREDICTIONS")
        print("=" * 80)

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
        bouts_df = cast(pd.DataFrame, bouts_df[bouts_df['basho_id'] >= start_basho])
    if end_basho is not None:
        bouts_df = cast(pd.DataFrame, bouts_df[bouts_df['basho_id'] <= end_basho])

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
        print("Feature engineer now has:")
        print(f"  - Elo ratings for {len(engineer.elo_system.ratings)} wrestlers")
        print(f"  - Historical stats for {len(engineer.rikishi_stats)} wrestlers")

    ensemble_weights = {'rf': 0.45, 'lgb': 0.45, 'xgb': 0.10}

    # Get optimized parameters
    if use_gpu and gpu_optimizer:
        recommendations = gpu_optimizer.get_training_recommendations()
        n_estimators = recommendations.get('recommended_n_estimators', 400)
    else:
        n_estimators = 400  # CPU-optimized default

    # LightGBM configuration
    lgb_base_params = {
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': n_estimators,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1,
    }
    if use_gpu and gpu_optimizer:
        lgb_params = gpu_optimizer.get_lightgbm_params(lgb_base_params)
    else:
        lgb_params = {**lgb_base_params, 'n_jobs': -1}  # CPU multi-threading

    # XGBoost configuration
    xgb_base_params = {
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': n_estimators,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
    }
    if use_gpu and gpu_optimizer:
        xgb_params = gpu_optimizer.get_xgboost_params(xgb_base_params)
    else:
        xgb_params = {**xgb_base_params, 'n_jobs': -1}  # CPU multi-threading

    # ---------------------------------------------------------------------------
    # Walk-forward validation
    #
    # Each bout produces exactly 2 samples in X (winner-as-A and loser-as-A),
    # so X has 2*N rows in chronological order.  We split at bout boundaries so
    # that both mirror samples always land in the same partition — preventing any
    # implicit leakage between symmetric pairs.
    #
    # Methodology: expanding-window, 5 non-overlapping test windows covering the
    # last 30% of bouts.  For each fold we:
    #   1. Train all three models on everything BEFORE the test window.
    #   2. Blend their predicted probabilities (not classify-then-average).
    #   3. Compute ensemble accuracy for that window.
    # This mirrors real production use: we always predict forward in time.
    # ---------------------------------------------------------------------------
    n_bouts = len(bouts_df)
    n_folds = 5
    eval_bouts = int(n_bouts * 0.30)        # last 30% used for evaluation
    train_bouts_base = n_bouts - eval_bouts  # first 70% always in training
    window_size = eval_bouts // n_folds      # bouts per test window

    if verbose:
        print(f"\nWalk-forward validation ({n_folds} folds, last {eval_bouts} bouts evaluated)...")
        print(f"  Base training size: {train_bouts_base} bouts, window size: {window_size} bouts each")

    fold_accuracies: list[float] = []
    rf_fold_accs: list[float] = []
    lgb_fold_accs: list[float] = []
    xgb_fold_accs: list[float] = []

    for fold in range(n_folds):
        # Bout index range for this test window
        test_start_bout = train_bouts_base + fold * window_size
        test_end_bout = test_start_bout + window_size

        # Convert to sample indices (2 samples per bout, interleaved in order)
        test_start_sample = test_start_bout * 2
        test_end_sample = test_end_bout * 2

        X_tr = X.iloc[:test_start_sample]
        y_tr = y.iloc[:test_start_sample]
        X_te = X.iloc[test_start_sample:test_end_sample]
        y_te = y.iloc[test_start_sample:test_end_sample]

        rf_fold = _make_rf().fit(X_tr, y_tr)
        lgb_fold = lgb.LGBMClassifier(**lgb_params).fit(X_tr, y_tr)
        xgb_fold = xgb.XGBClassifier(**xgb_params).fit(X_tr, y_tr)

        # Blend probabilities before thresholding — this is the actual ensemble
        proba = (
            ensemble_weights['rf'] * rf_fold.predict_proba(X_te)[:, 1]
            + ensemble_weights['lgb'] * np.asarray(lgb_fold.predict_proba(X_te))[:, 1]
            + ensemble_weights['xgb'] * xgb_fold.predict_proba(X_te)[:, 1]
        )
        fold_acc = float(accuracy_score(y_te, proba >= 0.5))
        fold_accuracies.append(fold_acc)

        # Individual model accuracies for reference only
        rf_fold_accs.append(float(accuracy_score(y_te, rf_fold.predict(X_te))))
        lgb_fold_accs.append(float(accuracy_score(
            y_te, np.asarray(lgb_fold.predict(X_te)).ravel()
        )))
        xgb_fold_accs.append(float(accuracy_score(y_te, xgb_fold.predict(X_te))))

        if verbose:
            n_train = test_start_bout
            print(f"  Fold {fold + 1}: train={n_train} bouts, test={window_size} bouts -> {fold_acc:.4f}")

    accuracy = float(np.mean(fold_accuracies))
    accuracy_std = float(np.std(fold_accuracies))

    if verbose:
        print("\nWalk-Forward Validation Results:")
        print(f"  Random Forest:  {np.mean(rf_fold_accs):.4f} (+/- {np.std(rf_fold_accs) * 2:.4f})")
        print(f"  LightGBM:       {np.mean(lgb_fold_accs):.4f} (+/- {np.std(lgb_fold_accs) * 2:.4f})")
        print(f"  XGBoost:        {np.mean(xgb_fold_accs):.4f} (+/- {np.std(xgb_fold_accs) * 2:.4f})")
        print(f"  Ensemble:       {accuracy:.4f} (+/- {accuracy_std * 2:.4f})")

    # Train final models on ALL data for production
    if verbose:
        print("\nTraining final models on ALL data for production...")

    if verbose:
        print("  [1/3] Random Forest...")
    rf_model = _make_rf()
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

    # Get project root and save to models directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, 'models')

    # Save XGBoost model using native format to avoid pickle compatibility issues
    xgb_booster_path = os.path.join(models_dir, 'xgboost_booster.json')
    if verbose:
        print(f"  Saving XGBoost booster to: {xgb_booster_path}")
    xgb_model.get_booster().save_model(xgb_booster_path)

    model_package = {
        'config': config,
        'feature_engineer': engineer,  # Contains Elo ratings, stats, etc.
        'models': {
            'random_forest': rf_model,
            'lightgbm': lgb_model,
            'xgboost': {'_type': 'xgboost_booster', 'path': 'models/xgboost_booster.json'},
        },
        'ensemble_weights': ensemble_weights,
        'feature_names': list(X.columns),
        'training_date': datetime.now().isoformat(),
        'num_training_bouts': len(bouts_df),
        'accuracy': accuracy,
        'last_trained_basho_id': latest_basho if latest_basho else end_basho,
        'last_trained_day': latest_day,
    }

    filename = os.path.join(models_dir, 'sumo_predictor_production.joblib')
    joblib.dump(model_package, filename)

    if verbose:
        print(f"\n[OK] Model saved to: {filename}")
        print("\nThis package contains:")
        print("  - Trained ensemble models (RF + LGB + XGB)")
        print("  - Feature engineer with Elo ratings and historical stats")
        print("  - Configuration and metadata")
        print(f"\nWalk-Forward Ensemble Accuracy: {accuracy * 100:.2f}% (+/- {accuracy_std * 2 * 100:.2f}%)")
        print("=" * 80)

    return {
        'num_training_bouts': len(bouts_df),
        'accuracy': accuracy,
        'model_path': filename,
    }


if __name__ == "__main__":
    train_and_save_production_model()
