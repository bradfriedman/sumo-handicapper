"""
Quick Iterative Improvement - Test key hyperparameters
"""
from iterative_improve import test_configuration, ModelConfig

# Test 4 key configurations
configs = [
    {'name': 'Baseline (K=32)', 'elo_k_factor': 32, 'recent_bouts_window': 10},
    {'name': 'Higher K-factor (K=40)', 'elo_k_factor': 40, 'recent_bouts_window': 10},
    {'name': 'Baseline + Larger Window', 'elo_k_factor': 32, 'recent_bouts_window': 15},
    {'name': 'High K + Large Window', 'elo_k_factor': 40, 'recent_bouts_window': 15},
]

print("="*80)
print("QUICK ITERATIVE IMPROVEMENT")
print("Testing 4 key configurations")
print("="*80)

results = []
best_acc = 0
best_config = None

for i, cfg in enumerate(configs, 1):
    print(f"\n{'#'*80}")
    print(f"Configuration {i}/4: {cfg['name']}")
    print(f"{'#'*80}")

    config = ModelConfig(
        elo_k_factor=cfg['elo_k_factor'],
        recent_bouts_window=cfg['recent_bouts_window'],
        include_head_to_head=True,
        include_age=True
    )

    result, predictor, X_test, y_test = test_configuration(config, i)

    if result and 'random_forest' in result['models']:
        acc = result['models']['random_forest']['accuracy']
        results.append((cfg['name'], acc, result['models']['random_forest']['roc_auc']))

        if acc > best_acc:
            best_acc = acc
            best_config = cfg['name']
            print(f"\n🎯 NEW BEST! Accuracy: {acc:.4f}")

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

for name, acc, auc in results:
    marker = " ⭐" if acc == best_acc else ""
    print(f"{name:30s} | Acc: {acc:.4f} | AUC: {auc:.4f}{marker}")

print(f"\nBest Configuration: {best_config}")
print(f"Best Accuracy: {best_acc:.4f}")
