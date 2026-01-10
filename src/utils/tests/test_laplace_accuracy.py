"""
Test accuracy of model with Laplace smoothing on basho win rate
"""
from enhanced_features import EnhancedFeatureEngineer
from sumo_predictor import ModelConfig, SumoDataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
import numpy as np

print("="*80)
print("TESTING MODEL ACCURACY WITH LAPLACE SMOOTHING")
print("="*80)

# Load data
print("\nLoading historical data...")
config = ModelConfig(elo_k_factor=32, recent_bouts_window=15)
loader = SumoDataLoader()
bouts_df = loader.load_raw_bouts()

# Build features with Laplace smoothing
print("Building features with Laplace smoothing...")
engineer = EnhancedFeatureEngineer(config)
X, y = engineer.build_dataset(bouts_df)

print(f"\nDataset size: {len(X)} samples")
print(f"Number of features: {len(X.columns)}")

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set:  {len(X_test)} samples")

# Train ensemble models
print("\nTraining ensemble models...")

print("  [1/3] Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"        Random Forest accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

print("  [2/3] LightGBM...")
lgb_model = lgb.LGBMClassifier(
    max_depth=6,
    learning_rate=0.03,
    n_estimators=400,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test))
print(f"        LightGBM accuracy: {lgb_acc:.4f} ({lgb_acc*100:.2f}%)")

print("  [3/3] XGBoost...")
xgb_model = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=400,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"        XGBoost accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")

# Ensemble prediction (weighted)
print("\n" + "="*80)
print("ENSEMBLE PREDICTION (45% RF + 45% LGB + 10% XGB)")
print("="*80)

rf_proba = rf_model.predict_proba(X_test)[:, 1]
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

ensemble_proba = 0.45 * rf_proba + 0.45 * lgb_proba + 0.10 * xgb_proba
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n🎯 FINAL ENSEMBLE ACCURACY: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
print("\nComparison with previous results:")
print("  - Before Laplace smoothing: 60.41%")
print(f"  - After Laplace smoothing:  {ensemble_acc*100:.2f}%")

if ensemble_acc > 0.6041:
    improvement = (ensemble_acc - 0.6041) * 100
    print(f"  ✅ Improvement: +{improvement:.2f} percentage points")
elif ensemble_acc < 0.6041:
    decrease = (0.6041 - ensemble_acc) * 100
    print(f"  ⚠️  Decrease: -{decrease:.2f} percentage points")
else:
    print("  ➖ No change in accuracy")

print("\n" + "="*80)
