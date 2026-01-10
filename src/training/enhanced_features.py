"""
Enhanced Feature Engineering with Momentum, Rank-Specific Performance, and Career Phase
Goal: Beat 60.37% ensemble accuracy
"""
import numpy as np
from src.core.sumo_predictor import (
    ModelConfig, SumoDataLoader, FeatureEngineer, EloRatingSystem
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from typing import Dict, Tuple, List
from collections import deque
from datetime import datetime
from tqdm import tqdm


class EnhancedFeatureEngineer(FeatureEngineer):
    """Extended feature engineering with momentum, rank-specific stats, and career phase"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Additional tracking for enhanced features
        self.rank_specific_stats = {}  # (rikishi_id) -> {vs_higher: {w, l}, vs_lower: {w, l}, vs_yokozuna: {w, l}}
        self.rank_history = {}  # (rikishi_id) -> [(basho_id, rank)]
        self.momentum_decay = 0.5  # Exponential decay factor

    def _init_rank_specific_stats(self, rikishi_id: int):
        """Initialize rank-specific statistics"""
        if rikishi_id not in self.rank_specific_stats:
            self.rank_specific_stats[rikishi_id] = {
                'vs_higher': {'wins': 0, 'losses': 0},
                'vs_lower': {'wins': 0, 'losses': 0},
                'vs_yokozuna': {'wins': 0, 'losses': 0},
                'vs_ozeki': {'wins': 0, 'losses': 0},
            }
        if rikishi_id not in self.rank_history:
            self.rank_history[rikishi_id] = []

    def _calculate_momentum(self, recent_bouts: List[int]) -> float:
        """
        Calculate exponentially weighted momentum
        More recent bouts have higher weight: weight = decay^age
        """
        if not recent_bouts:
            return 0.5

        weighted_sum = 0.0
        weight_sum = 0.0

        for age, result in enumerate(reversed(recent_bouts)):
            weight = self.momentum_decay ** age
            weighted_sum += result * weight
            weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else 0.5

    def _calculate_rank_trend(self, rikishi_id: int, current_basho: int, current_rank: int) -> Tuple[float, float]:
        """
        Calculate rank trend (improving/declining career phase)
        Returns: (short_term_trend, long_term_trend)

        IMPORTANT: Uses ONLY historical data, not current bout's rank
        """
        history = self.rank_history.get(rikishi_id, [])

        # DO NOT add current rank during feature extraction - that would be data leakage!
        # We only use historical ranks from BEFORE this bout

        if len(history) < 2:
            return 0.0, 0.0

        # Short-term trend (last 3 bashos) - using ONLY past data
        recent_history = history[-3:]
        if len(recent_history) >= 2:
            # Lower rank number = better, so trend DOWN is good
            short_trend = (recent_history[0][1] - recent_history[-1][1]) / len(recent_history)
        else:
            short_trend = 0.0

        # Long-term trend (last 6 bashos)
        if len(history) >= 6:
            long_trend = (history[-6][1] - history[-1][1]) / 6
        else:
            long_trend = short_trend

        return short_trend, long_trend

    def extract_features_for_bout(self, bout_row: pd.Series) -> Dict:
        """Enhanced feature extraction with momentum and rank-specific stats"""

        # Get base features from parent class
        features = super().extract_features_for_bout(bout_row)

        rikishi_a = bout_row['winning_rikishi_id']
        rikishi_b = bout_row['losing_rikishi_id']
        rank_a = bout_row['winner_rank']
        rank_b = bout_row['loser_rank']
        basho_id = bout_row['basho_id']

        # Initialize rank-specific tracking
        self._init_rank_specific_stats(rikishi_a)
        self._init_rank_specific_stats(rikishi_b)

        # 1. MOMENTUM with exponential decay
        recent_a = self.rikishi_stats[rikishi_a]['recent_bouts'][-20:]  # Use more history
        recent_b = self.rikishi_stats[rikishi_b]['recent_bouts'][-20:]

        features['rikishi_a_momentum'] = self._calculate_momentum(recent_a)
        features['rikishi_b_momentum'] = self._calculate_momentum(recent_b)
        features['momentum_diff'] = features['rikishi_a_momentum'] - features['rikishi_b_momentum']

        # 2. RANK-SPECIFIC PERFORMANCE
        stats_a = self.rank_specific_stats[rikishi_a]
        stats_b = self.rank_specific_stats[rikishi_b]

        # Performance vs higher ranked opponents
        vs_higher_a = stats_a['vs_higher']
        total_vs_higher_a = vs_higher_a['wins'] + vs_higher_a['losses']
        features['rikishi_a_vs_higher_rate'] = vs_higher_a['wins'] / total_vs_higher_a if total_vs_higher_a > 0 else 0.5

        vs_higher_b = stats_b['vs_higher']
        total_vs_higher_b = vs_higher_b['wins'] + vs_higher_b['losses']
        features['rikishi_b_vs_higher_rate'] = vs_higher_b['wins'] / total_vs_higher_b if total_vs_higher_b > 0 else 0.5

        # Performance vs lower ranked opponents
        vs_lower_a = stats_a['vs_lower']
        total_vs_lower_a = vs_lower_a['wins'] + vs_lower_a['losses']
        features['rikishi_a_vs_lower_rate'] = vs_lower_a['wins'] / total_vs_lower_a if total_vs_lower_a > 0 else 0.5

        vs_lower_b = stats_b['vs_lower']
        total_vs_lower_b = vs_lower_b['wins'] + vs_lower_b['losses']
        features['rikishi_b_vs_lower_rate'] = vs_lower_b['wins'] / total_vs_lower_b if total_vs_lower_b > 0 else 0.5

        # Performance vs yokozuna specifically
        vs_yoko_a = stats_a['vs_yokozuna']
        total_vs_yoko_a = vs_yoko_a['wins'] + vs_yoko_a['losses']
        features['rikishi_a_vs_yokozuna_rate'] = vs_yoko_a['wins'] / total_vs_yoko_a if total_vs_yoko_a > 0 else 0.5

        vs_yoko_b = stats_b['vs_yokozuna']
        total_vs_yoko_b = vs_yoko_b['wins'] + vs_yoko_b['losses']
        features['rikishi_b_vs_yokozuna_rate'] = vs_yoko_b['wins'] / total_vs_yoko_b if total_vs_yoko_b > 0 else 0.5

        # 3. CAREER PHASE / RANK TREND
        short_trend_a, long_trend_a = self._calculate_rank_trend(rikishi_a, basho_id, rank_a)
        short_trend_b, long_trend_b = self._calculate_rank_trend(rikishi_b, basho_id, rank_b)

        features['rikishi_a_short_trend'] = short_trend_a
        features['rikishi_b_short_trend'] = short_trend_b
        features['rikishi_a_long_trend'] = long_trend_a
        features['rikishi_b_long_trend'] = long_trend_b
        features['trend_diff'] = short_trend_a - short_trend_b

        # 4. FEATURE INTERACTIONS (most important)
        features['elo_rank_interaction'] = features['elo_diff'] * features['rank_diff']
        features['momentum_rank_interaction'] = features['momentum_diff'] * features['rank_diff']
        features['experience_age_interaction'] = features.get('experience_diff', 0) * (
            features.get('rikishi_a_age_years', 25) - features.get('rikishi_b_age_years', 25)
        )

        # 5. STREAK FEATURES
        # Win/loss streak
        streak_a = 0
        for result in reversed(self.rikishi_stats[rikishi_a]['recent_bouts'][-10:]):
            if (streak_a >= 0 and result == 1) or (streak_a < 0 and result == 0):
                streak_a += 1 if result == 1 else -1
            else:
                break

        streak_b = 0
        for result in reversed(self.rikishi_stats[rikishi_b]['recent_bouts'][-10:]):
            if (streak_b >= 0 and result == 1) or (streak_b < 0 and result == 0):
                streak_b += 1 if result == 1 else -1
            else:
                break

        features['rikishi_a_streak'] = streak_a
        features['rikishi_b_streak'] = streak_b
        features['streak_diff'] = streak_a - streak_b

        return features

    def build_dataset(self, bouts_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Build dataset with proper handling of enhanced diff features"""
        features_list = []
        labels = []

        print("Building features chronologically...")
        for idx, bout in tqdm(bouts_df.iterrows(), total=len(bouts_df)):
            # Extract features BEFORE updating
            features = self.extract_features_for_bout(bout)

            # Sample 1: winner is rikishi_a (label=1)
            features_list.append(features.copy())
            labels.append(1)

            # Sample 2: loser is rikishi_a (label=0)
            # Swap all a/b features AND properly handle enhanced diff features
            swapped_features = {}
            for key, value in features.items():
                if key.startswith('rikishi_a_'):
                    new_key = key.replace('rikishi_a_', 'rikishi_b_')
                    swapped_features[new_key] = value
                elif key.startswith('rikishi_b_'):
                    new_key = key.replace('rikishi_b_', 'rikishi_a_')
                    swapped_features[new_key] = value
                elif key in ['elo_diff', 'rank_diff', 'experience_diff', 'momentum_diff',
                             'trend_diff', 'streak_diff']:
                    # All diff features must be negated when swapping
                    swapped_features[key] = -value
                elif key in ['elo_rank_interaction', 'momentum_rank_interaction',
                             'experience_age_interaction']:
                    # Interaction features: diff1 * diff2 => (-diff1) * (-diff2) = diff1 * diff2
                    # So interactions of two diffs should STAY THE SAME, not be negated!
                    swapped_features[key] = value
                elif key == 'h2h_a_wins':
                    swapped_features['h2h_b_wins'] = value
                elif key == 'h2h_b_wins':
                    swapped_features['h2h_a_wins'] = value
                elif key == 'h2h_a_win_rate':
                    swapped_features[key] = 1 - value if value != 0.5 else 0.5
                else:
                    # day_of_basho and other symmetric features stay the same
                    swapped_features[key] = value

            features_list.append(swapped_features)
            labels.append(0)

            # NOW update statistics for next bout
            self.update_after_bout(bout, bout['winning_rikishi_id'], bout['losing_rikishi_id'])

        X = pd.DataFrame(features_list)
        y = pd.Series(labels, name='winner')

        print(f"\nDataset shape (before cleaning): {X.shape}")
        print(f"Features: {list(X.columns)}")
        print(f"Label distribution: {y.value_counts().to_dict()}")

        # Check for NaN values
        nan_counts = X.isna().sum()
        if nan_counts.any():
            print(f"\nNaN values found:")
            print(nan_counts[nan_counts > 0])
            print("\nFilling NaN values...")
            for col in X.columns:
                if 'age' in col:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)

        print(f"\nFinal dataset shape: {X.shape}")
        print(f"NaN check after filling: {X.isna().sum().sum()} NaN values remain")

        return X, y

    def update_after_bout(self, bout_row: pd.Series, winner_id: int, loser_id: int):
        """Update statistics including rank-specific performance"""

        # Update base stats
        super().update_after_bout(bout_row, winner_id, loser_id)

        # Update rank-specific stats
        winner_rank = bout_row['winner_rank']
        loser_rank = bout_row['loser_rank']
        basho_id = bout_row['basho_id']

        self._init_rank_specific_stats(winner_id)
        self._init_rank_specific_stats(loser_id)

        # Update rank history AFTER bout is processed (avoid data leakage)
        # Only add if this is a new basho for this rikishi
        if not self.rank_history[winner_id] or self.rank_history[winner_id][-1][0] != basho_id:
            self.rank_history[winner_id].append((basho_id, winner_rank))

        if not self.rank_history[loser_id] or self.rank_history[loser_id][-1][0] != basho_id:
            self.rank_history[loser_id].append((basho_id, loser_rank))

        # Winner stats
        if winner_rank > loser_rank:  # Winner beat higher ranked opponent
            self.rank_specific_stats[winner_id]['vs_higher']['wins'] += 1
        elif winner_rank < loser_rank:  # Winner beat lower ranked opponent
            self.rank_specific_stats[winner_id]['vs_lower']['wins'] += 1

        if loser_rank == -3:  # Beat a yokozuna
            self.rank_specific_stats[winner_id]['vs_yokozuna']['wins'] += 1
        elif loser_rank == -2:  # Beat an ozeki
            self.rank_specific_stats[winner_id]['vs_ozeki']['wins'] += 1

        # Loser stats
        if loser_rank > winner_rank:  # Loser lost to lower ranked opponent
            self.rank_specific_stats[loser_id]['vs_lower']['losses'] += 1
        elif loser_rank < winner_rank:  # Loser lost to higher ranked opponent
            self.rank_specific_stats[loser_id]['vs_higher']['losses'] += 1

        if winner_rank == -3:  # Lost to a yokozuna
            self.rank_specific_stats[loser_id]['vs_yokozuna']['losses'] += 1
        elif winner_rank == -2:  # Lost to an ozeki
            self.rank_specific_stats[loser_id]['vs_ozeki']['losses'] += 1


def main():
    """Train and test enhanced features"""
    print("="*80)
    print("ENHANCED FEATURE ENGINEERING")
    print("="*80)
    print("\nNew features:")
    print("  1. Exponentially weighted momentum")
    print("  2. Rank-specific performance (vs higher/lower/yokozuna)")
    print("  3. Career phase (rank trending up/down)")
    print("  4. Feature interactions (elo×rank, momentum×rank)")
    print("  5. Win/loss streaks")
    print("\nBaseline to beat: 60.37% (Ensemble)")
    print("="*80)

    # Load data
    print("\nLoading data...")
    config = ModelConfig(elo_k_factor=32, recent_bouts_window=15)
    loader = SumoDataLoader()
    bouts_df = loader.load_raw_bouts()

    # Build ENHANCED features
    print("\nBuilding enhanced features...")
    engineer = EnhancedFeatureEngineer(config)
    X, y = engineer.build_dataset(bouts_df)

    print(f"\nEnhanced dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Added features: {X.shape[1] - 21} new features")

    # Split data (same random state for fair comparison)
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n" + "="*80)
    print("TRAINING MODELS WITH ENHANCED FEATURES")
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

    # 4. Ensemble
    print("\n[4/4] Ensemble (45% RF + 45% LGB + 10% XGB)...")
    ensemble_proba = 0.45 * rf_proba + 0.45 * lgb_proba + 0.10 * xgb_proba
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    print(f"   Accuracy: {ensemble_acc:.4f}, ROC-AUC: {ensemble_auc:.4f}")
    results.append(('Ensemble', ensemble_acc, ensemble_auc))

    # Summary
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    print(f"\n{'Model':<20}{'Accuracy':<12}{'ROC-AUC':<12}{'vs Baseline':<15}{'Status'}")
    print("-" * 80)
    print(f"{'[Baseline] Old Ens.':<20}{'0.6037':<12}{'0.6538':<12}{'+0.00%':<15}{'Baseline'}")
    print("-" * 80)

    best_acc = 0.0
    best_model = None

    for name, acc, auc in results:
        diff = (acc - 0.6037) * 100
        status = "🎉 NEW BEST!" if acc > 0.6037 else ("✓ Matched" if acc == 0.6037 else "")
        print(f"{name:<20}{acc:<12.4f}{auc:<12.4f}{diff:+.2f}%{'':<10}{status}")
        if acc > best_acc:
            best_acc = acc
            best_model = name

    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    if best_acc > 0.6037:
        improvement = (best_acc - 0.6037) * 100
        print(f"\n🎉 SUCCESS! Enhanced features improved the model!")
        print(f"Best Model: {best_model}")
        print(f"Accuracy: {best_acc:.4f} ({improvement:+.2f}% improvement)")
        print(f"\nApproximate additional correct predictions: {int(16288 * improvement / 100)}")

        # Feature importance
        if 'LightGBM' in best_model or 'Random Forest' in best_model:
            model = lgb_model if 'LightGBM' in best_model else rf_model
            if hasattr(model, 'feature_importances_'):
                print("\nTop 15 Most Important Features:")
                importances = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                print(importances.head(15).to_string(index=False))
    else:
        print(f"\n❌ Enhanced features did not improve accuracy.")
        print(f"Best enhanced: {best_acc:.4f}")
        print(f"Baseline: 0.6037")
        print(f"\nPossible reasons:")
        print("  - New features may be too noisy")
        print("  - Already near the ceiling for this data")
        print("  - May need more hyperparameter tuning for new features")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
