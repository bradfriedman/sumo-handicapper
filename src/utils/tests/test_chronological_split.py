"""
Test enhanced features with CHRONOLOGICAL split instead of random split
This prevents the model from learning from future data
"""
from enhanced_features import EnhancedFeatureEngineer
from sumo_predictor import ModelConfig, SumoDataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import pandas as pd

print("="*80)
print("TESTING WITH CHRONOLOGICAL TRAIN/TEST SPLIT")
print("="*80)
print("\nThis tests whether the high accuracy is due to random split data leakage")
print("Chronological split: Train on older bouts, test on newer bouts")
print("\n" + "="*80)

# Load data
print("\nLoading data...")
config = ModelConfig(elo_k_factor=32, recent_bouts_window=15)
loader = SumoDataLoader()
bouts_df = loader.load_raw_bouts()

# Build enhanced features (chronologically)
print("\nBuilding enhanced features...")
engineer = EnhancedFeatureEngineer(config)
X, y = engineer.build_dataset(bouts_df)

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")

# CHRONOLOGICAL split: first 80% for training, last 20% for testing
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Chronological split:")
print(f"  Train: {len(X_train)} samples (first 80%)")
print(f"  Test:  {len(X_test)} samples (last 20%)")

print("\n" + "="*80)
print("TRAINING WITH CHRONOLOGICAL SPLIT")
print("="*80)

results = []

# 1. Random Forest
print("\n[1/3] Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)
print(f"   Accuracy: {rf_acc:.4f}, ROC-AUC: {rf_auc:.4f}")
results.append(('Random Forest', rf_acc, rf_auc))

# 2. LightGBM
print("\n[2/3] LightGBM...")
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
lgb_pred = lgb_model.predict(X_test)
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_acc = accuracy_score(y_test, lgb_pred)
lgb_auc = roc_auc_score(y_test, lgb_proba)
print(f"   Accuracy: {lgb_acc:.4f}, ROC-AUC: {lgb_auc:.4f}")
results.append(('LightGBM', lgb_acc, lgb_auc))

# 3. XGBoost
print("\n[3/3] XGBoost...")
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
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_proba)
print(f"   Accuracy: {xgb_acc:.4f}, ROC-AUC: {xgb_auc:.4f}")
results.append(('XGBoost', xgb_acc, xgb_auc))

# Summary
print("\n" + "="*80)
print("RESULTS: CHRONOLOGICAL vs RANDOM SPLIT")
print("="*80)

print(f"\n{'Model':<20}{'Chronological':<15}{'Random (prev)':<15}{'Difference'}")
print("-" * 70)

# Previous random split results
random_results = {
    'Random Forest': 0.9509,
    'LightGBM': 0.9764,
    'XGBoost': 0.9453
}

for name, chrono_acc, chrono_auc in results:
    random_acc = random_results.get(name, 0)
    diff = (chrono_acc - random_acc) * 100
    print(f"{name:<20}{chrono_acc:<15.4f}{random_acc:<15.4f}{diff:+.2f}%")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("\nIf chronological accuracy is MUCH LOWER than random split:")
print("  → Random split was leaking future information to past predictions")
print("  → Chronological split is the TRUE model performance")
print("\nIf chronological accuracy is SIMILAR to random split:")
print("  → There may be other data leakage issues in feature engineering")
print("\n" + "="*80)
