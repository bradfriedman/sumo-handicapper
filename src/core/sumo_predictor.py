"""
Sumo Bout Prediction Model with Iterative Improvement
"""
import pymysql
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List
from dataclasses import dataclass
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from src.core.db_connector import get_connection, get_connection_params

# Try importing XGBoost (requires libomp on Mac)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"Warning: XGBoost not available - install libomp: brew install libomp")
    XGBOOST_AVAILABLE = False
    xgb = None

# Try importing LightGBM (also requires libomp on Mac)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    print(f"Warning: LightGBM not available - install libomp: brew install libomp")
    LIGHTGBM_AVAILABLE = False
    lgb = None


@dataclass
class ModelConfig:
    """Configurable hyperparameters for models"""
    # Elo parameters
    elo_k_factor: float = 32
    elo_initial_rating: float = 1500

    # Feature engineering
    recent_bouts_window: int = 10
    include_head_to_head: bool = True
    include_age: bool = True

    # Model hyperparameters - XGBoost
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 200
    xgb_min_child_weight: int = 1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # Model hyperparameters - LightGBM
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.1
    lgb_n_estimators: int = 200
    lgb_num_leaves: int = 31
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8

    # Train/test split
    test_size: float = 0.2
    random_state: int = 42


class EloRatingSystem:
    """Calculate and maintain Elo ratings for wrestlers"""

    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[int, float] = {}
        self.rating_history: List[Tuple[int, int, int, float, float]] = []  # basho, day, rikishi_id, rating

    def get_rating(self, rikishi_id: int) -> float:
        """Get current rating for a wrestler"""
        if rikishi_id not in self.ratings:
            self.ratings[rikishi_id] = self.initial_rating
        return self.ratings[rikishi_id]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for wrestler A against wrestler B"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, winner_id: int, loser_id: int, basho_id: int, day: int):
        """Update ratings after a bout"""
        winner_rating = self.get_rating(winner_id)
        loser_rating = self.get_rating(loser_id)

        expected_winner = self.expected_score(winner_rating, loser_rating)
        expected_loser = self.expected_score(loser_rating, winner_rating)

        new_winner_rating = winner_rating + self.k_factor * (1 - expected_winner)
        new_loser_rating = loser_rating + self.k_factor * (0 - expected_loser)

        # Store history before updating
        self.rating_history.append((basho_id, day, winner_id, winner_rating, loser_rating))
        self.rating_history.append((basho_id, day, loser_id, loser_rating, winner_rating))

        self.ratings[winner_id] = new_winner_rating
        self.ratings[loser_id] = new_loser_rating


class SumoDataLoader:
    """Load and preprocess sumo bout data"""

    def __init__(self, host='127.0.0.1', port=3307, user='dewsweeper',
                 password='dewsweeper_password123', database='dewsweeper3'):
        """
        Initialize the data loader.

        Note: Connection parameters are stored for backward compatibility only.
        The actual database connection uses db_connector.get_connection(), which
        checks environment variables in the following priority:
        1. CLOUD_SQL_CONNECTION_NAME (if set, uses Cloud SQL Python Connector)
        2. DB_HOST/DB_PORT (if set, uses proxy connection)
        3. Falls back to these constructor parameters

        Args:
            host: Database host (default: 127.0.0.1)
            port: Database port (default: 3307)
            user: Database user (default: dewsweeper)
            password: Database password
            database: Database name (default: dewsweeper3)
        """
        # Store connection parameters for backward compatibility
        # However, if CLOUD_SQL_CONNECTION_NAME is set, it will be used instead
        self.conn_params = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database
        }

    def load_raw_bouts(self) -> pd.DataFrame:
        """Load all valid bouts from database"""
        # Use new connection method that supports both proxy and Cloud SQL Connector
        conn = get_connection()

        query = """
        SELECT
            b.id as bout_id,
            b.basho_id,
            b.day,
            b.winning_rikishi_id,
            b.losing_rikishi_id,
            b.winning_rikishi_rank as winner_rank,
            b.losing_rikishi_rank as loser_rank,
            b.kimarite_id,
            k.name as kimarite_name,
            b.value,
            wr.real_name as winner_name,
            wr.dob as winner_dob,
            lr.real_name as loser_name,
            lr.dob as loser_dob
        FROM boi_ozumobout b
        JOIN boi_kimarite k ON b.kimarite_id = k.id
        JOIN boi_rikishi wr ON b.winning_rikishi_id = wr.id
        JOIN boi_rikishi lr ON b.losing_rikishi_id = lr.id
        WHERE k.name NOT LIKE '%hansoku%'
          AND k.name NOT LIKE '%default%'
          AND k.name NOT LIKE '%fusen%'
        ORDER BY b.basho_id, b.day, b.id
        """

        df = pd.read_sql(query, conn)
        conn.close()

        print(f"Loaded {len(df)} valid bouts")
        return df


class FeatureEngineer:
    """Engineer features for bout prediction"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.elo_system = EloRatingSystem(
            k_factor=config.elo_k_factor,
            initial_rating=config.elo_initial_rating
        )

        # Track statistics
        self.rikishi_stats: Dict[int, Dict] = {}
        self.head_to_head: Dict[Tuple[int, int], Dict] = {}
        self.basho_records: Dict[Tuple[int, int], Dict] = {}  # (rikishi_id, basho_id) -> stats

        # Database connection for live queries (optional, for prediction time)
        self.db_config = None

    def _init_rikishi_stats(self, rikishi_id: int):
        """Initialize stats for a wrestler"""
        if rikishi_id not in self.rikishi_stats:
            self.rikishi_stats[rikishi_id] = {
                'total_bouts': 0,
                'total_wins': 0,
                'recent_bouts': [],
                'bashos': set()
            }

    def _init_basho_record(self, rikishi_id: int, basho_id: int):
        """Initialize record for a wrestler in a specific basho"""
        key = (rikishi_id, basho_id)
        if key not in self.basho_records:
            self.basho_records[key] = {'wins': 0, 'losses': 0}

    def _get_head_to_head(self, rikishi_a: int, rikishi_b: int) -> Tuple[int, int]:
        """Get head-to-head record (a_wins, b_wins)"""
        key1 = (rikishi_a, rikishi_b)
        key2 = (rikishi_b, rikishi_a)

        if key1 in self.head_to_head:
            return self.head_to_head[key1]['a_wins'], self.head_to_head[key1]['b_wins']
        elif key2 in self.head_to_head:
            return self.head_to_head[key2]['b_wins'], self.head_to_head[key2]['a_wins']
        else:
            return 0, 0

    def _update_head_to_head(self, winner_id: int, loser_id: int):
        """Update head-to_head record"""
        key1 = (winner_id, loser_id)
        key2 = (loser_id, winner_id)

        if key1 in self.head_to_head:
            self.head_to_head[key1]['a_wins'] += 1
        elif key2 in self.head_to_head:
            self.head_to_head[key2]['b_wins'] += 1
        else:
            self.head_to_head[key1] = {'a_wins': 1, 'b_wins': 0}

    def enable_live_data(self, db_config: Dict):
        """Enable live database queries for fresh data at prediction time"""
        self.db_config = db_config

    def _get_live_head_to_head(self, rikishi_a: int, rikishi_b: int) -> Tuple[int, int]:
        """Query fresh head-to-head record from database"""
        if not self.db_config:
            return self._get_head_to_head(rikishi_a, rikishi_b)

        conn = get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                COALESCE(SUM(CASE WHEN winning_rikishi_id = %s THEN 1 ELSE 0 END), 0) as a_wins,
                COALESCE(SUM(CASE WHEN winning_rikishi_id = %s THEN 1 ELSE 0 END), 0) as b_wins
            FROM boi_ozumobout
            WHERE (winning_rikishi_id = %s AND losing_rikishi_id = %s)
               OR (winning_rikishi_id = %s AND losing_rikishi_id = %s)
        """
        cursor.execute(query, (rikishi_a, rikishi_b, rikishi_a, rikishi_b, rikishi_b, rikishi_a))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        return (int(result[0]), int(result[1]))

    def _get_live_recent_bouts(self, rikishi_id: int, basho_id: int, limit: int = None) -> List[int]:
        """Query fresh recent bout results from database"""
        if not self.db_config:
            # Fall back to cached data
            if rikishi_id in self.rikishi_stats:
                recent = self.rikishi_stats[rikishi_id]['recent_bouts']
                if limit:
                    return recent[-limit:]
                return recent
            return []

        if limit is None:
            limit = self.config.recent_bouts_window

        conn = get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                CASE WHEN winning_rikishi_id = %s THEN 1 ELSE 0 END as won
            FROM boi_ozumobout
            WHERE (winning_rikishi_id = %s OR losing_rikishi_id = %s)
              AND basho_id < %s
              AND kimarite_id NOT IN (
                  SELECT id FROM boi_kimarite
                  WHERE name IN ('hansoku', 'default', 'fusen')
              )
            ORDER BY basho_id DESC, day DESC
            LIMIT %s
        """
        cursor.execute(query, (rikishi_id, rikishi_id, rikishi_id, basho_id, limit))
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return [r[0] for r in reversed(results)]

    def _get_live_basho_record(self, rikishi_id: int, basho_id: int, day: int) -> Tuple[int, int]:
        """Query fresh basho record from database (before the specified day)"""
        if not self.db_config:
            # Fall back to cached data
            key = (rikishi_id, basho_id)
            if key in self.basho_records:
                rec = self.basho_records[key]
                return rec['wins'], rec['losses']
            return 0, 0

        conn = get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                COALESCE(SUM(CASE WHEN winning_rikishi_id = %s THEN 1 ELSE 0 END), 0) as wins,
                COALESCE(SUM(CASE WHEN losing_rikishi_id = %s THEN 1 ELSE 0 END), 0) as losses
            FROM boi_ozumobout
            WHERE (winning_rikishi_id = %s OR losing_rikishi_id = %s)
              AND basho_id = %s
              AND day < %s
              AND kimarite_id NOT IN (
                  SELECT id FROM boi_kimarite
                  WHERE name IN ('hansoku', 'default', 'fusen')
              )
        """
        cursor.execute(query, (rikishi_id, rikishi_id, rikishi_id, rikishi_id, basho_id, day))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        return (int(result[0]), int(result[1]))

    def extract_features_for_bout(self, bout_row: pd.Series) -> Dict:
        """Extract features for a single bout before it happens"""
        rikishi_a = bout_row['winning_rikishi_id']
        rikishi_b = bout_row['losing_rikishi_id']
        basho_id = bout_row['basho_id']
        day = bout_row['day']

        self._init_rikishi_stats(rikishi_a)
        self._init_rikishi_stats(rikishi_b)
        self._init_basho_record(rikishi_a, basho_id)
        self._init_basho_record(rikishi_b, basho_id)

        features = {}

        # Elo ratings (before the bout)
        features['rikishi_a_elo'] = self.elo_system.get_rating(rikishi_a)
        features['rikishi_b_elo'] = self.elo_system.get_rating(rikishi_b)
        features['elo_diff'] = features['rikishi_a_elo'] - features['rikishi_b_elo']

        # Rank features
        features['rikishi_a_rank'] = bout_row['winner_rank']
        features['rikishi_b_rank'] = bout_row['loser_rank']
        features['rank_diff'] = bout_row['winner_rank'] - bout_row['loser_rank']

        # Experience
        features['rikishi_a_total_bouts'] = self.rikishi_stats[rikishi_a]['total_bouts']
        features['rikishi_b_total_bouts'] = self.rikishi_stats[rikishi_b]['total_bouts']
        features['experience_diff'] = features['rikishi_a_total_bouts'] - features['rikishi_b_total_bouts']

        # Overall win rate
        a_stats = self.rikishi_stats[rikishi_a]
        b_stats = self.rikishi_stats[rikishi_b]
        features['rikishi_a_win_rate'] = a_stats['total_wins'] / a_stats['total_bouts'] if a_stats['total_bouts'] > 0 else 0.5
        features['rikishi_b_win_rate'] = b_stats['total_wins'] / b_stats['total_bouts'] if b_stats['total_bouts'] > 0 else 0.5

        # Recent form (last N bouts) - use live data if available
        if self.db_config:
            recent_a = self._get_live_recent_bouts(rikishi_a, basho_id)
            recent_b = self._get_live_recent_bouts(rikishi_b, basho_id)
        else:
            recent_a = a_stats['recent_bouts'][-self.config.recent_bouts_window:]
            recent_b = b_stats['recent_bouts'][-self.config.recent_bouts_window:]
        features['rikishi_a_recent_win_rate'] = sum(recent_a) / len(recent_a) if recent_a else 0.5
        features['rikishi_b_recent_win_rate'] = sum(recent_b) / len(recent_b) if recent_b else 0.5

        # Current basho record with Laplace smoothing - use live data if available
        if self.db_config:
            basho_a_wins, basho_a_losses = self._get_live_basho_record(rikishi_a, basho_id, day)
            basho_b_wins, basho_b_losses = self._get_live_basho_record(rikishi_b, basho_id, day)
        else:
            basho_a = self.basho_records[(rikishi_a, basho_id)]
            basho_b = self.basho_records[(rikishi_b, basho_id)]
            basho_a_wins, basho_a_losses = basho_a['wins'], basho_a['losses']
            basho_b_wins, basho_b_losses = basho_b['wins'], basho_b['losses']

        # Apply Laplace smoothing (+1 win, +1 loss pseudo-counts)
        features['rikishi_a_basho_win_rate'] = (basho_a_wins + 1) / (basho_a_wins + basho_a_losses + 2)
        features['rikishi_b_basho_win_rate'] = (basho_b_wins + 1) / (basho_b_wins + basho_b_losses + 2)

        # Head-to-head - use live data if available
        if self.config.include_head_to_head:
            if self.db_config:
                a_h2h_wins, b_h2h_wins = self._get_live_head_to_head(rikishi_a, rikishi_b)
            else:
                a_h2h_wins, b_h2h_wins = self._get_head_to_head(rikishi_a, rikishi_b)
            features['h2h_a_wins'] = a_h2h_wins
            features['h2h_b_wins'] = b_h2h_wins
            h2h_total = a_h2h_wins + b_h2h_wins
            features['h2h_a_win_rate'] = a_h2h_wins / h2h_total if h2h_total > 0 else 0.5

        # Contextual
        features['day_of_basho'] = day

        # Age (if available and configured)
        if self.config.include_age and pd.notna(bout_row['winner_dob']) and pd.notna(bout_row['loser_dob']):
            # Approximate age at time of bout (using basho_id as rough date proxy)
            # Each basho is roughly 2 months, starting from some base year
            features['rikishi_a_age_years'] = (datetime.now() - pd.to_datetime(bout_row['winner_dob'])).days / 365.25
            features['rikishi_b_age_years'] = (datetime.now() - pd.to_datetime(bout_row['loser_dob'])).days / 365.25

        return features

    def update_after_bout(self, bout_row: pd.Series, winner_id: int, loser_id: int):
        """Update all statistics after a bout completes"""
        basho_id = bout_row['basho_id']
        day = bout_row['day']

        # Update Elo
        self.elo_system.update_ratings(winner_id, loser_id, basho_id, day)

        # Update overall stats
        self.rikishi_stats[winner_id]['total_bouts'] += 1
        self.rikishi_stats[winner_id]['total_wins'] += 1
        self.rikishi_stats[winner_id]['recent_bouts'].append(1)
        self.rikishi_stats[winner_id]['bashos'].add(basho_id)

        self.rikishi_stats[loser_id]['total_bouts'] += 1
        self.rikishi_stats[loser_id]['recent_bouts'].append(0)
        self.rikishi_stats[loser_id]['bashos'].add(basho_id)

        # Update basho records
        self.basho_records[(winner_id, basho_id)]['wins'] += 1
        self.basho_records[(loser_id, basho_id)]['losses'] += 1

        # Update head-to-head
        if self.config.include_head_to_head:
            self._update_head_to_head(winner_id, loser_id)

    def build_dataset(self, bouts_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Build feature matrix and labels from bout data"""
        features_list = []
        labels = []

        print("Building features chronologically...")
        for idx, bout in tqdm(bouts_df.iterrows(), total=len(bouts_df)):
            # Extract features BEFORE updating (so we predict future from past)
            features = self.extract_features_for_bout(bout)

            # Create two samples: one for each wrestler as "rikishi_a"
            # Sample 1: winner is rikishi_a (label=1)
            features_list.append(features.copy())
            labels.append(1)

            # Sample 2: loser is rikishi_a (label=0)
            # Swap all a/b features explicitly
            swapped_features = {}
            for key, value in features.items():
                if key.startswith('rikishi_a_'):
                    new_key = key.replace('rikishi_a_', 'rikishi_b_')
                    swapped_features[new_key] = value
                elif key.startswith('rikishi_b_'):
                    new_key = key.replace('rikishi_b_', 'rikishi_a_')
                    swapped_features[new_key] = value
                elif key in ['elo_diff', 'rank_diff', 'experience_diff']:
                    swapped_features[key] = -value
                elif key == 'h2h_a_wins':
                    swapped_features['h2h_b_wins'] = value
                elif key == 'h2h_b_wins':
                    swapped_features['h2h_a_wins'] = value
                elif key == 'h2h_a_win_rate':
                    # When swapping, b's win rate is 1 - a's win rate
                    swapped_features[key] = 1 - value if value != 0.5 else 0.5
                else:
                    # day_of_basho stays the same
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
            # Fill NaN values appropriately
            # For age features, use median
            for col in X.columns:
                if 'age' in col:
                    X[col] = X[col].fillna(X[col].median())
                # For other numeric features, use 0 or mean
                else:
                    X[col] = X[col].fillna(0)

        print(f"\nFinal dataset shape: {X.shape}")
        print(f"NaN check after filling: {X.isna().sum().sum()} NaN values remain")

        return X, y


class SumoPredictor:
    """Train and evaluate sumo bout prediction models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.feature_names = None
        self.results = {}

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train multiple models"""
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)

        self.feature_names = X_train.columns.tolist()

        # Logistic Regression (baseline)
        print("\n1. Training Logistic Regression...")
        self.models['logistic'] = LogisticRegression(
            max_iter=1000,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        self.models['logistic'].fit(X_train, y_train)
        print("   ✓ Logistic Regression trained")

        # XGBoost
        if XGBOOST_AVAILABLE:
            print("\n2. Training XGBoost...")
            self.models['xgboost'] = xgb.XGBClassifier(
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                n_estimators=self.config.xgb_n_estimators,
                min_child_weight=self.config.xgb_min_child_weight,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                random_state=self.config.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
            self.models['xgboost'].fit(X_train, y_train)
            print("   ✓ XGBoost trained")
        else:
            print("\n2. Skipping XGBoost (not available)")

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n3. Training LightGBM...")
            self.models['lightgbm'] = lgb.LGBMClassifier(
                max_depth=self.config.lgb_max_depth,
                learning_rate=self.config.lgb_learning_rate,
                n_estimators=self.config.lgb_n_estimators,
                num_leaves=self.config.lgb_num_leaves,
                subsample=self.config.lgb_subsample,
                colsample_bytree=self.config.lgb_colsample_bytree,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=-1
            )
            self.models['lightgbm'].fit(X_train, y_train)
            print("   ✓ LightGBM trained")
        else:
            print("\n3. Skipping LightGBM (not available)")

        # Random Forest (fallback if gradient boosting not available)
        print("\n4. Training Random Forest...")
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X_train, y_train)
        print("   ✓ Random Forest trained")

        print("\nAll models trained successfully!")

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate all trained models"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)

        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC:  {roc_auc:.4f}")

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Loser', 'Winner']))

            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)

            self.results[model_name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': cm
            }

        return self.results

    def analyze_errors(self, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = 'lightgbm'):
        """Analyze prediction errors to find patterns"""
        print("\n" + "="*80)
        print(f"ERROR ANALYSIS - {model_name.upper()}")
        print("="*80)

        model = self.models[model_name]
        y_pred = model.predict(X_test)

        # Find incorrect predictions
        errors = X_test[y_test != y_pred].copy()
        errors['true_label'] = y_test[y_test != y_pred]
        errors['predicted_label'] = y_pred[y_test != y_pred]

        print(f"\nTotal errors: {len(errors)} out of {len(X_test)} ({len(errors)/len(X_test)*100:.2f}%)")

        if len(errors) > 0:
            print("\nError analysis by feature ranges:")

            # Analyze errors by Elo difference
            print("\n1. Elo Difference Distribution in Errors:")
            print(errors['elo_diff'].describe())

            # Analyze errors by rank difference
            print("\n2. Rank Difference Distribution in Errors:")
            print(errors['rank_diff'].describe())

            # Analyze errors by experience
            print("\n3. Experience Difference Distribution in Errors:")
            print(errors['experience_diff'].describe())

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                print("\n4. Top 10 Most Important Features:")
                importances = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                print(importances.head(10).to_string(index=False))

        return errors

    def save_models(self, filepath: str = 'sumo_models.joblib'):
        """Save trained models"""
        joblib.dump({
            'models': self.models,
            'config': self.config,
            'feature_names': self.feature_names
        }, filepath)
        print(f"\nModels saved to {filepath}")


def main():
    """Main training pipeline"""
    print("="*80)
    print("SUMO BOUT PREDICTION MODEL")
    print("="*80)

    # Configuration
    config = ModelConfig()

    # Load data
    print("\n1. Loading data...")
    loader = SumoDataLoader()
    bouts_df = loader.load_raw_bouts()

    # Build features
    print("\n2. Engineering features...")
    engineer = FeatureEngineer(config)
    X, y = engineer.build_dataset(bouts_df)

    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    # Train models
    print("\n4. Training models...")
    predictor = SumoPredictor(config)
    predictor.train_models(X_train, y_train)

    # Evaluate
    print("\n5. Evaluating models...")
    results = predictor.evaluate_models(X_test, y_test)

    # Error analysis
    print("\n6. Analyzing errors...")
    # Use best available model for error analysis
    if 'lightgbm' in predictor.models:
        best_model = 'lightgbm'
    elif 'xgboost' in predictor.models:
        best_model = 'xgboost'
    elif 'random_forest' in predictor.models:
        best_model = 'random_forest'
    else:
        best_model = 'logistic'
    errors = predictor.analyze_errors(X_test, y_test, model_name=best_model)

    # Save models
    predictor.save_models()

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)

    return predictor, results, errors


if __name__ == '__main__':
    predictor, results, errors = main()
