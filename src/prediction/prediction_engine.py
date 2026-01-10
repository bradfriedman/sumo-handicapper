"""
Shared prediction engine for sumo bout predictions
Used by both CLI scripts and Streamlit UI
"""
import joblib
import pandas as pd
import pymysql
import sys
import os
from src.core.fantasy_points import calculate_expected_points, get_rank_label

# Database configuration
# Supports both local development (Cloud SQL Proxy) and Streamlit Cloud (direct connection)
# In Streamlit Cloud, set secrets in the dashboard under Settings > Secrets
try:
    import streamlit as st
    # Try to get config from Streamlit secrets first
    DB_CONFIG = {
        'host': st.secrets.get('DB_HOST', '127.0.0.1'),
        'port': int(st.secrets.get('DB_PORT', 3307)),
        'user': st.secrets.get('DB_USER', 'dewsweeper'),
        'password': st.secrets.get('DB_PASSWORD', 'dewsweeper_password123'),
        'database': st.secrets.get('DB_NAME', 'dewsweeper3')
    }
except (ImportError, FileNotFoundError):
    # Fallback to environment variables or defaults (for CLI usage)
    DB_CONFIG = {
        'host': os.environ.get('DB_HOST', '127.0.0.1'),
        'port': int(os.environ.get('DB_PORT', '3307')),
        'user': os.environ.get('DB_USER', 'dewsweeper'),
        'password': os.environ.get('DB_PASSWORD', 'dewsweeper_password123'),
        'database': os.environ.get('DB_NAME', 'dewsweeper3')
    }

def get_db_connection():
    """Get database connection"""
    return pymysql.connect(**DB_CONFIG)

def load_model():
    """Load the trained model package"""
    # Get the project root directory (2 levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, 'models', 'sumo_predictor_production.joblib')

    try:
        model_package = joblib.load(model_path)

        # Enable live data queries for fresh stats at prediction time
        model_package['feature_engineer'].enable_live_data(DB_CONFIG)

        return model_package
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at '{model_path}'. "
                              "Please run 'python3 src/training/save_best_model.py' first.")

def search_rikishi_by_name(name_query, basho_id=None):
    """
    Search for rikishi by name (partial match)
    Searches both real_name AND shikona (ring name)
    Returns list of matching rikishi with their info
    """
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # Search by both real_name AND shikona (ring name)
    if basho_id:
        query = """
            SELECT DISTINCT
                r.id,
                r.real_name,
                r.dob,
                e.shikona as ring_name
            FROM boi_rikishi r
            LEFT JOIN boi_ozumobanzukeentry e ON r.id = e.rikishi_id AND e.basho_id = %s
            WHERE r.real_name LIKE %s OR e.shikona LIKE %s
            ORDER BY r.real_name
        """
        cursor.execute(query, (basho_id, f'%{name_query}%', f'%{name_query}%'))
    else:
        query = """
            SELECT DISTINCT
                r.id,
                r.real_name,
                r.dob,
                e.shikona as ring_name
            FROM boi_rikishi r
            LEFT JOIN boi_ozumobanzukeentry e ON r.id = e.rikishi_id
            WHERE r.real_name LIKE %s OR e.shikona LIKE %s
            ORDER BY r.real_name
            LIMIT 20
        """
        cursor.execute(query, (f'%{name_query}%', f'%{name_query}%'))

    results = cursor.fetchall()

    # If basho_id provided, get rank for that basho
    if basho_id and results:
        for rikishi in results:
            rank_query = """
                SELECT `rank`
                FROM boi_ozumobanzukeentry
                WHERE rikishi_id = %s AND basho_id = %s
                LIMIT 1
            """
            cursor.execute(rank_query, (rikishi['id'], basho_id))
            rank_result = cursor.fetchone()
            rikishi['rank'] = rank_result['rank'] if rank_result else None
            rikishi['basho_id'] = basho_id

    cursor.close()
    conn.close()

    return results

def get_rikishi_by_id(rikishi_id, basho_id=None):
    """Get rikishi details by ID"""
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    if basho_id:
        query = """
            SELECT
                r.id,
                r.real_name,
                r.dob,
                e.shikona as ring_name,
                e.rank
            FROM boi_rikishi r
            LEFT JOIN boi_ozumobanzukeentry e ON r.id = e.rikishi_id AND e.basho_id = %s
            WHERE r.id = %s
        """
        cursor.execute(query, (basho_id, rikishi_id))
    else:
        query = """
            SELECT
                r.id,
                r.real_name,
                r.dob
            FROM boi_rikishi r
            WHERE r.id = %s
        """
        cursor.execute(query, (rikishi_id,))

    result = cursor.fetchone()
    cursor.close()
    conn.close()

    return result

def predict_bout(model_package, rikishi_a_id, rikishi_b_id, basho_id, day,
                 rikishi_a_rank=None, rikishi_b_rank=None,
                 rikishi_a_dob=None, rikishi_b_dob=None):
    """
    Predict the outcome of a bout between two rikishi

    Returns:
        dict with prediction results
    """
    engineer = model_package['feature_engineer']
    models = model_package['models']
    weights = model_package['ensemble_weights']

    # Create a minimal bout row for feature extraction
    bout_data = {
        'winning_rikishi_id': rikishi_a_id,
        'losing_rikishi_id': rikishi_b_id,
        'basho_id': basho_id,
        'day': day,
        'winner_rank': rikishi_a_rank if rikishi_a_rank else 1,
        'loser_rank': rikishi_b_rank if rikishi_b_rank else 1,
        'winner_dob': rikishi_a_dob,
        'loser_dob': rikishi_b_dob
    }

    bout_series = pd.Series(bout_data)

    # Extract features
    try:
        features = engineer.extract_features_for_bout(bout_series)
    except Exception as e:
        return {
            'error': f"Failed to extract features: {str(e)}",
            'suggestion': "Make sure both rikishi have historical data in the database"
        }

    # Convert to DataFrame (single row)
    X = pd.DataFrame([features])

    # Ensure all expected features are present
    for feat in model_package['feature_names']:
        if feat not in X.columns:
            X[feat] = 0

    # Reorder columns to match training
    X = X[model_package['feature_names']]

    # Get predictions from each model
    rf_proba = models['random_forest'].predict_proba(X)[0, 1]
    lgb_proba = models['lightgbm'].predict_proba(X)[0, 1]
    xgb_proba = models['xgboost'].predict_proba(X)[0, 1]

    # Ensemble prediction
    ensemble_proba = (
        weights['rf'] * rf_proba +
        weights['lgb'] * lgb_proba +
        weights['xgb'] * xgb_proba
    )

    # Determine winner
    predicted_winner = rikishi_a_id if ensemble_proba >= 0.5 else rikishi_b_id
    confidence = ensemble_proba if ensemble_proba >= 0.5 else 1 - ensemble_proba

    # Calculate fantasy points
    exp_a, exp_b, pot_a, pot_b = calculate_expected_points(
        rikishi_a_rank, rikishi_b_rank, ensemble_proba, 1 - ensemble_proba
    )

    return {
        'rikishi_a_id': rikishi_a_id,
        'rikishi_b_id': rikishi_b_id,
        'rikishi_a_win_probability': ensemble_proba,
        'rikishi_b_win_probability': 1 - ensemble_proba,
        'predicted_winner_id': predicted_winner,
        'confidence': confidence,
        'individual_predictions': {
            'random_forest': rf_proba,
            'lightgbm': lgb_proba,
            'xgboost': xgb_proba
        },
        'key_features': {
            'elo_diff': features.get('elo_diff', 'N/A'),
            'rank_diff': features.get('rank_diff', 'N/A'),
            'experience_diff': features.get('experience_diff', 'N/A'),
            'momentum_diff': features.get('momentum_diff', 'N/A')
        },
        'head_to_head': {
            'rikishi_a_wins': features.get('h2h_a_wins', 0),
            'rikishi_b_wins': features.get('h2h_b_wins', 0),
            'rikishi_a_win_rate': features.get('h2h_a_win_rate', 0.5)
        },
        'fantasy_points': {
            'rikishi_a_expected': exp_a,
            'rikishi_b_expected': exp_b,
            'rikishi_a_potential': pot_a,
            'rikishi_b_potential': pot_b,
            'rikishi_a_rank': rikishi_a_rank,
            'rikishi_b_rank': rikishi_b_rank
        }
    }
