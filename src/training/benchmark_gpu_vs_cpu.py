"""
Benchmark GPU vs CPU Training Performance

Compares training time and accuracy between:
1. Original CPU-only training (n_jobs=-1)
2. New GPU-optimized training (device=cuda)
"""
import os
import sys
import time
from src.training.enhanced_features import EnhancedFeatureEngineer
from src.core.sumo_predictor import ModelConfig, SumoDataLoader
from src.utils.gpu_optimizer import GPUOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import numpy as np


def format_time(seconds):
    """Format seconds as human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def train_cpu_only(X, y, n_estimators=400):
    """Train models using original CPU-only approach"""
    print("\n" + "="*80)
    print("TRAINING WITH CPU ONLY (Original Approach)")
    print("="*80)

    results = {}

    # Random Forest
    print("\n[1/3] Training Random Forest (CPU)...")
    start = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Original: CPU multi-threading
    )
    rf_model.fit(X, y)
    rf_time = time.time() - start
    rf_acc = accuracy_score(y, rf_model.predict(X))
    results['rf'] = {'time': rf_time, 'accuracy': rf_acc}
    print(f"  Time: {format_time(rf_time)}, Accuracy: {rf_acc:.4f}")

    # LightGBM
    print("\n[2/3] Training LightGBM (CPU)...")
    start = time.time()
    lgb_model = lgb.LGBMClassifier(
        max_depth=6,
        learning_rate=0.03,
        n_estimators=n_estimators,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,  # Original: CPU multi-threading
        verbose=-1
    )
    lgb_model.fit(X, y)
    lgb_time = time.time() - start
    lgb_acc = accuracy_score(y, lgb_model.predict(X))
    results['lgb'] = {'time': lgb_time, 'accuracy': lgb_acc}
    print(f"  Time: {format_time(lgb_time)}, Accuracy: {lgb_acc:.4f}")

    # XGBoost
    print("\n[3/3] Training XGBoost (CPU)...")
    start = time.time()
    xgb_model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=n_estimators,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,  # Original: CPU multi-threading
        eval_metric='logloss'
    )
    xgb_model.fit(X, y)
    xgb_time = time.time() - start
    xgb_acc = accuracy_score(y, xgb_model.predict(X))
    results['xgb'] = {'time': xgb_time, 'accuracy': xgb_acc}
    print(f"  Time: {format_time(xgb_time)}, Accuracy: {xgb_acc:.4f}")

    total_time = rf_time + lgb_time + xgb_time
    results['total'] = {'time': total_time}

    print(f"\nTotal CPU Training Time: {format_time(total_time)}")

    return results


def train_gpu_optimized(X, y, gpu_optimizer):
    """Train models using new GPU-optimized approach"""
    print("\n" + "="*80)
    print("TRAINING WITH GPU OPTIMIZATION (New Approach)")
    print("="*80)

    results = {}

    recommendations = gpu_optimizer.get_training_recommendations()
    n_estimators = recommendations.get('recommended_n_estimators', 400)

    print(f"\nUsing n_estimators={n_estimators} (GPU-optimized)")

    # Random Forest (stays on CPU)
    print("\n[1/3] Training Random Forest (CPU)...")
    start = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)
    rf_time = time.time() - start
    rf_acc = accuracy_score(y, rf_model.predict(X))
    results['rf'] = {'time': rf_time, 'accuracy': rf_acc}
    print(f"  Time: {format_time(rf_time)}, Accuracy: {rf_acc:.4f}")

    # LightGBM with GPU params (if available)
    print("\n[2/3] Training LightGBM (GPU if available)...")
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
    lgb_params = gpu_optimizer.get_lightgbm_params(lgb_base_params)

    start = time.time()
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X, y)
    lgb_time = time.time() - start
    lgb_acc = accuracy_score(y, lgb_model.predict(X))
    results['lgb'] = {'time': lgb_time, 'accuracy': lgb_acc}
    print(f"  Time: {format_time(lgb_time)}, Accuracy: {lgb_acc:.4f}")

    # XGBoost with GPU params
    print("\n[3/3] Training XGBoost (GPU if available)...")
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
    xgb_params = gpu_optimizer.get_xgboost_params(xgb_base_params)

    start = time.time()
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X, y)
    xgb_time = time.time() - start
    xgb_acc = accuracy_score(y, xgb_model.predict(X))
    results['xgb'] = {'time': xgb_time, 'accuracy': xgb_acc}
    print(f"  Time: {format_time(xgb_time)}, Accuracy: {xgb_acc:.4f}")

    total_time = rf_time + lgb_time + xgb_time
    results['total'] = {'time': total_time}

    print(f"\nTotal GPU-Optimized Training Time: {format_time(total_time)}")

    return results


def print_comparison(cpu_results, gpu_results):
    """Print comparison table"""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)

    print("\n{:<15} {:<15} {:<15} {:<15}".format("Model", "CPU Time", "GPU Time", "Speedup"))
    print("-" * 80)

    for model in ['rf', 'lgb', 'xgb']:
        cpu_time = cpu_results[model]['time']
        gpu_time = gpu_results[model]['time']
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

        print("{:<15} {:<15} {:<15} {:<15}".format(
            model.upper(),
            format_time(cpu_time),
            format_time(gpu_time),
            f"{speedup:.2f}x"
        ))

    print("-" * 80)
    cpu_total = cpu_results['total']['time']
    gpu_total = gpu_results['total']['time']
    total_speedup = cpu_total / gpu_total if gpu_total > 0 else 1.0

    print("{:<15} {:<15} {:<15} {:<15}".format(
        "TOTAL",
        format_time(cpu_total),
        format_time(gpu_total),
        f"{speedup:.2f}x"
    ))

    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)

    print("\n{:<15} {:<20} {:<20} {:<15}".format("Model", "CPU Accuracy", "GPU Accuracy", "Difference"))
    print("-" * 80)

    for model in ['rf', 'lgb', 'xgb']:
        cpu_acc = cpu_results[model]['accuracy']
        gpu_acc = gpu_results[model]['accuracy']
        diff = gpu_acc - cpu_acc

        print("{:<15} {:<20} {:<20} {:<15}".format(
            model.upper(),
            f"{cpu_acc:.4f}",
            f"{gpu_acc:.4f}",
            f"{diff:+.4f}"
        ))

    print("\n" + "="*80)

    # Summary
    time_saved = cpu_total - gpu_total
    time_saved_pct = (time_saved / cpu_total * 100) if cpu_total > 0 else 0

    print("\nSUMMARY:")
    if total_speedup > 1.0:
        print(f"  GPU optimization is {total_speedup:.2f}x faster than CPU-only")
        print(f"  Time saved: {format_time(time_saved)} ({time_saved_pct:.1f}%)")
    elif total_speedup < 1.0:
        print(f"  CPU-only is {1/total_speedup:.2f}x faster (GPU overhead not worth it for this dataset)")
    else:
        print(f"  Performance is similar between CPU and GPU")

    print("="*80)


def main():
    """Run benchmark"""
    print("="*80)
    print("GPU vs CPU TRAINING BENCHMARK")
    print("="*80)

    # Initialize GPU optimizer
    print("\nInitializing GPU detection...")
    gpu_optimizer = GPUOptimizer()
    gpu_optimizer.print_summary()

    # Load data
    print("\nLoading data...")
    config = ModelConfig(elo_k_factor=32, recent_bouts_window=15)
    loader = SumoDataLoader()
    bouts_df = loader.load_raw_bouts()

    # Build features
    print("Building features...")
    engineer = EnhancedFeatureEngineer(config)
    X, y = engineer.build_dataset(bouts_df)

    print(f"\nDataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Training on {len(bouts_df)} bouts")

    # Run CPU benchmark
    cpu_results = train_cpu_only(X, y, n_estimators=400)

    # Run GPU benchmark
    gpu_results = train_gpu_optimized(X, y, gpu_optimizer)

    # Print comparison
    print_comparison(cpu_results, gpu_results)

    # Save results
    import json
    from datetime import datetime

    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(X),
        'num_bouts': len(bouts_df),
        'num_features': len(X.columns),
        'gpu_name': gpu_optimizer.gpu_name,
        'gpu_memory_mb': gpu_optimizer.gpu_memory_mb,
        'cuda_available': gpu_optimizer.cuda_available,
        'cpu_results': {k: {'time': v['time'], 'accuracy': v.get('accuracy', 0)}
                       for k, v in cpu_results.items()},
        'gpu_results': {k: {'time': v['time'], 'accuracy': v.get('accuracy', 0)}
                       for k, v in gpu_results.items()}
    }

    output_file = 'benchmark_gpu_vs_cpu_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
