"""
Ensemble Modeling: Combining Random Forest + LightGBM + XGBoost
Goal: Beat 60.17% accuracy
"""
from sumo_predictor import ModelConfig, SumoDataLoader, FeatureEngineer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import joblib
from datetime import datetime

print("="*80)
print("ENSEMBLE MODEL EXPERIMENTATION")
print("="*80)
print("\nTarget to beat: 60.17% (Random Forest / LightGBM)")
print("\n" + "="*80)

# Load and prepare data ONCE
print("\nLoading data...")
config = ModelConfig(elo_k_factor=32, recent_bouts_window=15)
loader = SumoDataLoader()
bouts_df = loader.load_raw_bouts()

print("Building features...")
engineer = FeatureEngineer(config)
X, y = engineer.build_dataset(bouts_df)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Define our best models
print("\n" + "="*80)
print("TRAINING INDIVIDUAL MODELS")
print("="*80)

print("\n1. Training Random Forest (baseline winner)...")
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
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred_proba)
print(f"   Random Forest: {rf_acc:.4f} accuracy, {rf_auc:.4f} ROC-AUC")

print("\n2. Training LightGBM (best tuned)...")
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
lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_acc = accuracy_score(y_test, lgb_pred)
lgb_auc = roc_auc_score(y_test, lgb_pred_proba)
print(f"   LightGBM: {lgb_acc:.4f} accuracy, {lgb_auc:.4f} ROC-AUC")

print("\n3. Training XGBoost (best tuned)...")
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
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
print(f"   XGBoost: {xgb_acc:.4f} accuracy, {xgb_auc:.4f} ROC-AUC")

# Ensemble strategies
print("\n" + "="*80)
print("TESTING ENSEMBLE STRATEGIES")
print("="*80)

results = []
best_accuracy = 0.6017
best_ensemble = None
best_name = None

# Strategy 1: Simple Voting (Hard Voting)
print("\n[1/7] Hard Voting Ensemble (majority vote)...")
hard_voting = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('lgb', lgb_model),
        ('xgb', xgb_model)
    ],
    voting='hard'
)
# Already fitted, just need to predict
hard_votes = np.array([rf_pred, lgb_pred, xgb_pred])
hard_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=hard_votes)
hard_acc = accuracy_score(y_test, hard_pred)
print(f"   Accuracy: {hard_acc:.4f}")
results.append(('Hard Voting', hard_acc, None))
if hard_acc > best_accuracy:
    best_accuracy = hard_acc
    best_name = 'Hard Voting'
    print(f"   🎯 NEW BEST! +{(hard_acc - 0.6017)*100:.2f}%")

# Strategy 2: Soft Voting (Average probabilities)
print("\n[2/7] Soft Voting Ensemble (average probabilities)...")
soft_proba = (rf_pred_proba + lgb_pred_proba + xgb_pred_proba) / 3
soft_pred = (soft_proba >= 0.5).astype(int)
soft_acc = accuracy_score(y_test, soft_pred)
soft_auc = roc_auc_score(y_test, soft_proba)
print(f"   Accuracy: {soft_acc:.4f}, ROC-AUC: {soft_auc:.4f}")
results.append(('Soft Voting (Equal)', soft_acc, soft_auc))
if soft_acc > best_accuracy:
    best_accuracy = soft_acc
    best_name = 'Soft Voting (Equal)'
    best_ensemble = ('soft', [1/3, 1/3, 1/3])
    print(f"   🎯 NEW BEST! +{(soft_acc - 0.6017)*100:.2f}%")

# Strategy 3: Weighted by individual accuracy
print("\n[3/7] Weighted Ensemble (by accuracy)...")
total_acc = rf_acc + lgb_acc + xgb_acc
w_rf = rf_acc / total_acc
w_lgb = lgb_acc / total_acc
w_xgb = xgb_acc / total_acc
weighted_proba = w_rf * rf_pred_proba + w_lgb * lgb_pred_proba + w_xgb * xgb_pred_proba
weighted_pred = (weighted_proba >= 0.5).astype(int)
weighted_acc = accuracy_score(y_test, weighted_pred)
weighted_auc = roc_auc_score(y_test, weighted_proba)
print(f"   Weights: RF={w_rf:.3f}, LGB={w_lgb:.3f}, XGB={w_xgb:.3f}")
print(f"   Accuracy: {weighted_acc:.4f}, ROC-AUC: {weighted_auc:.4f}")
results.append(('Weighted by Accuracy', weighted_acc, weighted_auc))
if weighted_acc > best_accuracy:
    best_accuracy = weighted_acc
    best_name = 'Weighted by Accuracy'
    best_ensemble = ('weighted', [w_rf, w_lgb, w_xgb])
    print(f"   🎯 NEW BEST! +{(weighted_acc - 0.6017)*100:.2f}%")

# Strategy 4: Weighted by ROC-AUC
print("\n[4/7] Weighted Ensemble (by ROC-AUC)...")
total_auc = rf_auc + lgb_auc + xgb_auc
w_rf_auc = rf_auc / total_auc
w_lgb_auc = lgb_auc / total_auc
w_xgb_auc = xgb_auc / total_auc
weighted_auc_proba = w_rf_auc * rf_pred_proba + w_lgb_auc * lgb_pred_proba + w_xgb_auc * xgb_pred_proba
weighted_auc_pred = (weighted_auc_proba >= 0.5).astype(int)
weighted_auc_acc = accuracy_score(y_test, weighted_auc_pred)
weighted_auc_auc = roc_auc_score(y_test, weighted_auc_proba)
print(f"   Weights: RF={w_rf_auc:.3f}, LGB={w_lgb_auc:.3f}, XGB={w_xgb_auc:.3f}")
print(f"   Accuracy: {weighted_auc_acc:.4f}, ROC-AUC: {weighted_auc_auc:.4f}")
results.append(('Weighted by AUC', weighted_auc_acc, weighted_auc_auc))
if weighted_auc_acc > best_accuracy:
    best_accuracy = weighted_auc_acc
    best_name = 'Weighted by AUC'
    best_ensemble = ('weighted_auc', [w_rf_auc, w_lgb_auc, w_xgb_auc])
    print(f"   🎯 NEW BEST! +{(weighted_auc_acc - 0.6017)*100:.2f}%")

# Strategy 5: RF + LGB only (best two)
print("\n[5/7] RF + LGB Ensemble (best 2 models)...")
rf_lgb_proba = (rf_pred_proba + lgb_pred_proba) / 2
rf_lgb_pred = (rf_lgb_proba >= 0.5).astype(int)
rf_lgb_acc = accuracy_score(y_test, rf_lgb_pred)
rf_lgb_auc = roc_auc_score(y_test, rf_lgb_proba)
print(f"   Accuracy: {rf_lgb_acc:.4f}, ROC-AUC: {rf_lgb_auc:.4f}")
results.append(('RF + LGB Only', rf_lgb_acc, rf_lgb_auc))
if rf_lgb_acc > best_accuracy:
    best_accuracy = rf_lgb_acc
    best_name = 'RF + LGB Only'
    best_ensemble = ('rf_lgb', [0.5, 0.5])
    print(f"   🎯 NEW BEST! +{(rf_lgb_acc - 0.6017)*100:.2f}%")

# Strategy 6: Optimized weights (give more to best models)
print("\n[6/7] Optimized Weights (more to RF/LGB)...")
opt_proba = 0.45 * rf_pred_proba + 0.45 * lgb_pred_proba + 0.10 * xgb_pred_proba
opt_pred = (opt_proba >= 0.5).astype(int)
opt_acc = accuracy_score(y_test, opt_pred)
opt_auc = roc_auc_score(y_test, opt_proba)
print(f"   Weights: RF=0.45, LGB=0.45, XGB=0.10")
print(f"   Accuracy: {opt_acc:.4f}, ROC-AUC: {opt_auc:.4f}")
results.append(('Optimized 45-45-10', opt_acc, opt_auc))
if opt_acc > best_accuracy:
    best_accuracy = opt_acc
    best_name = 'Optimized 45-45-10'
    best_ensemble = ('optimized', [0.45, 0.45, 0.10])
    print(f"   🎯 NEW BEST! +{(opt_acc - 0.6017)*100:.2f}%")

# Strategy 7: Stacking with Logistic Regression
print("\n[7/7] Stacking Ensemble (meta-learner)...")
# Create meta-features from predictions
meta_features_train = np.column_stack([
    rf_model.predict_proba(X_train)[:, 1],
    lgb_model.predict_proba(X_train)[:, 1],
    xgb_model.predict_proba(X_train)[:, 1]
])
meta_features_test = np.column_stack([
    rf_pred_proba,
    lgb_pred_proba,
    xgb_pred_proba
])

meta_learner = LogisticRegression(max_iter=1000, random_state=42)
meta_learner.fit(meta_features_train, y_train)
stack_pred = meta_learner.predict(meta_features_test)
stack_proba = meta_learner.predict_proba(meta_features_test)[:, 1]
stack_acc = accuracy_score(y_test, stack_pred)
stack_auc = roc_auc_score(y_test, stack_proba)
print(f"   Meta-learner weights: RF={meta_learner.coef_[0][0]:.3f}, LGB={meta_learner.coef_[0][1]:.3f}, XGB={meta_learner.coef_[0][2]:.3f}")
print(f"   Accuracy: {stack_acc:.4f}, ROC-AUC: {stack_auc:.4f}")
results.append(('Stacking (LogReg)', stack_acc, stack_auc))
if stack_acc > best_accuracy:
    best_accuracy = stack_acc
    best_name = 'Stacking (LogReg)'
    best_ensemble = ('stacking', meta_learner)
    print(f"   🎯 NEW BEST! +{(stack_acc - 0.6017)*100:.2f}%")

# Summary
print("\n" + "="*80)
print("ENSEMBLE RESULTS SUMMARY")
print("="*80)

print(f"\n{'Strategy':<30}{'Accuracy':<12}{'ROC-AUC':<12}{'vs Baseline'}")
print("-" * 80)

# Add baseline
print(f"{'[Baseline] Single RF':<30}{rf_acc:<12.4f}{rf_auc:<12.4f}{'+0.00%'}")
print(f"{'[Baseline] Single LGB':<30}{lgb_acc:<12.4f}{lgb_auc:<12.4f}{'+0.00%'}")
print("-" * 80)

for name, acc, auc in results:
    diff = (acc - 0.6017) * 100
    marker = " ⭐" if acc >= 0.6017 else ""
    auc_str = f"{auc:.4f}" if auc else "N/A"
    print(f"{name:<30}{acc:<12.4f}{auc_str:<12}{diff:+.2f}%{marker}")

print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if best_accuracy > 0.6017:
    improvement = (best_accuracy - 0.6017) * 100
    print(f"\n🎉 SUCCESS! Ensemble beats individual models!")
    print(f"Best Ensemble: {best_name}")
    print(f"Accuracy: {best_accuracy:.4f} ({improvement:+.2f}% improvement)")

    # Save best ensemble
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_file = f"best_ensemble_{timestamp}.joblib"
    joblib.dump({
        'models': {
            'random_forest': rf_model,
            'lightgbm': lgb_model,
            'xgboost': xgb_model
        },
        'ensemble_type': best_ensemble[0] if best_ensemble else None,
        'ensemble_params': best_ensemble[1] if len(best_ensemble) > 1 else None,
        'accuracy': best_accuracy,
        'name': best_name,
        'feature_names': X.columns.tolist()
    }, ensemble_file)
    print(f"\nBest ensemble saved to: {ensemble_file}")
else:
    print(f"\n❌ No ensemble beat the baseline of 60.17%")
    print(f"Best ensemble: {best_name} with {best_accuracy:.4f}")
    print("\nIndividual models (RF or LGB) remain the best choice.")
    print("The models are already very strong and hard to improve with ensembling.")

print("\n" + "="*80)
