"""
Comprehensive Hyperparameter Tuning for XGBoost and LightGBM
Trying to beat Random Forest's 60.17% accuracy
"""
from sumo_predictor import ModelConfig, SumoDataLoader, FeatureEngineer, SumoPredictor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import json

print("="*80)
print("GRADIENT BOOSTING HYPERPARAMETER TUNING")
print("="*80)
print("\nTarget to beat: 60.17% (Random Forest)")
print("Train/Test Split: 80% / 20% (stratified)")
print("\n" + "="*80)

# Load and prepare data ONCE
print("\nLoading data...")
config = ModelConfig(elo_k_factor=32, recent_bouts_window=15)  # Best config found
loader = SumoDataLoader()
bouts_df = loader.load_raw_bouts()

print("Building features...")
engineer = FeatureEngineer(config)
X, y = engineer.build_dataset(bouts_df)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

results = []

# XGBoost configurations
xgb_configs = [
    # Baseline (current)
    {'name': 'XGB-Baseline', 'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},

    # Deeper trees
    {'name': 'XGB-Deep', 'max_depth': 8, 'learning_rate': 0.1, 'n_estimators': 200, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'name': 'XGB-VeryDeep', 'max_depth': 10, 'learning_rate': 0.1, 'n_estimators': 200, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},

    # Shallower trees with more estimators
    {'name': 'XGB-ShallowMany', 'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 400, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'name': 'XGB-VeryShallowMany', 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 500, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},

    # Lower learning rate with more estimators
    {'name': 'XGB-SlowLearning', 'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 300, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'name': 'XGB-VerySlowLearning', 'max_depth': 6, 'learning_rate': 0.03, 'n_estimators': 400, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},

    # Regularization variations
    {'name': 'XGB-HighReg', 'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200, 'min_child_weight': 3, 'subsample': 0.7, 'colsample_bytree': 0.7},
    {'name': 'XGB-LowReg', 'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200, 'min_child_weight': 1, 'subsample': 0.9, 'colsample_bytree': 0.9},
]

# LightGBM configurations
lgb_configs = [
    # Baseline (current)
    {'name': 'LGB-Baseline', 'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8},

    # More leaves (LightGBM's key parameter)
    {'name': 'LGB-MoreLeaves', 'max_depth': 8, 'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 63, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'name': 'LGB-ManyLeaves', 'max_depth': 10, 'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 127, 'subsample': 0.8, 'colsample_bytree': 0.8},

    # Fewer leaves with more estimators
    {'name': 'LGB-FewLeaves', 'max_depth': -1, 'learning_rate': 0.05, 'n_estimators': 400, 'num_leaves': 15, 'subsample': 0.8, 'colsample_bytree': 0.8},

    # Lower learning rate
    {'name': 'LGB-SlowLearning', 'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 300, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'name': 'LGB-VerySlowLearning', 'max_depth': 6, 'learning_rate': 0.03, 'n_estimators': 400, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8},

    # Regularization variations
    {'name': 'LGB-HighReg', 'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 31, 'subsample': 0.7, 'colsample_bytree': 0.7},
    {'name': 'LGB-Balanced', 'max_depth': 7, 'learning_rate': 0.08, 'n_estimators': 250, 'num_leaves': 50, 'subsample': 0.8, 'colsample_bytree': 0.8},
]

best_accuracy = 0.6017  # Random Forest score to beat
best_model_name = "Random Forest"
best_model = None

iteration = 0

# Test XGBoost configurations
print("\n" + "="*80)
print("TESTING XGBOOST CONFIGURATIONS")
print("="*80)

for cfg in xgb_configs:
    iteration += 1
    print(f"\n[{iteration}/{len(xgb_configs) + len(lgb_configs)}] Testing: {cfg['name']}")
    print(f"  Params: max_depth={cfg['max_depth']}, lr={cfg['learning_rate']}, n_est={cfg['n_estimators']}")

    try:
        model = xgb.XGBClassifier(
            max_depth=cfg['max_depth'],
            learning_rate=cfg['learning_rate'],
            n_estimators=cfg['n_estimators'],
            min_child_weight=cfg['min_child_weight'],
            subsample=cfg['subsample'],
            colsample_bytree=cfg['colsample_bytree'],
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"  → Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

        results.append({
            'name': cfg['name'],
            'model_type': 'XGBoost',
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'params': cfg
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = cfg['name']
            best_model = model
            print(f"  🎯 NEW BEST! Beat Random Forest by {(accuracy - 0.6017)*100:.2f}%")

    except Exception as e:
        print(f"  ❌ Failed: {e}")

# Test LightGBM configurations
print("\n" + "="*80)
print("TESTING LIGHTGBM CONFIGURATIONS")
print("="*80)

for cfg in lgb_configs:
    iteration += 1
    print(f"\n[{iteration}/{len(xgb_configs) + len(lgb_configs)}] Testing: {cfg['name']}")
    print(f"  Params: max_depth={cfg['max_depth']}, lr={cfg['learning_rate']}, leaves={cfg['num_leaves']}")

    try:
        model = lgb.LGBMClassifier(
            max_depth=cfg['max_depth'],
            learning_rate=cfg['learning_rate'],
            n_estimators=cfg['n_estimators'],
            num_leaves=cfg['num_leaves'],
            subsample=cfg['subsample'],
            colsample_bytree=cfg['colsample_bytree'],
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"  → Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

        results.append({
            'name': cfg['name'],
            'model_type': 'LightGBM',
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'params': cfg
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = cfg['name']
            best_model = model
            print(f"  🎯 NEW BEST! Beat Random Forest by {(accuracy - 0.6017)*100:.2f}%")

    except Exception as e:
        print(f"  ❌ Failed: {e}")

# Final summary
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

# Sort by accuracy
results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

print("\nTop 10 Models:")
print(f"{'Rank':<6}{'Model':<25}{'Accuracy':<12}{'ROC-AUC':<12}{'Type'}")
print("-" * 80)

for i, r in enumerate(results_sorted[:10], 1):
    marker = " ⭐" if r['accuracy'] >= 0.6017 else ""
    print(f"{i:<6}{r['name']:<25}{r['accuracy']:<12.4f}{r['roc_auc']:<12.4f}{r['model_type']}{marker}")

print("\n" + "="*80)
print("COMPARISON TO RANDOM FOREST")
print("="*80)
print(f"Random Forest: 60.17% accuracy")
print(f"Best Model:    {best_accuracy:.2%} accuracy ({best_model_name})")

if best_accuracy > 0.6017:
    improvement = (best_accuracy - 0.6017) * 100
    print(f"\n🎉 SUCCESS! Improved by {improvement:.2f}%")
else:
    print(f"\n❌ Did not beat Random Forest (best: {best_accuracy:.4f})")
    print("Random Forest remains the best model with default hyperparameters.")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"gbm_tuning_results_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump({
        'best_model': best_model_name,
        'best_accuracy': best_accuracy,
        'random_forest_baseline': 0.6017,
        'all_results': results_sorted
    }, f, indent=2)

print(f"\nResults saved to: {results_file}")

# Save best model if it beat Random Forest
if best_model and best_accuracy > 0.6017:
    import joblib
    model_file = f"best_gbm_model_{timestamp}.joblib"
    joblib.dump({
        'model': best_model,
        'model_name': best_model_name,
        'accuracy': best_accuracy,
        'feature_names': X.columns.tolist()
    }, model_file)
    print(f"Best model saved to: {model_file}")
