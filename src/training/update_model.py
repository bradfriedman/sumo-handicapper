"""
Incremental Model Update Script

Updates the prediction model with new bout data without retraining from scratch.
Tracks the last trained basho_id and day, then only processes new data.
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.training.save_best_model import train_and_save_production_model
from src.core.sumo_predictor import SumoDataLoader
import joblib
from tqdm import tqdm

# Training state file
STATE_FILE = os.path.join(project_root, '.model_training_state.json')

def load_training_state():
    """Load the last training state"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass

    # Default state - start from scratch
    return {
        'last_trained_basho_id': None,
        'last_trained_day': None,
        'last_training_date': None,
        'num_training_bouts': 0,
        'accuracy': None
    }

def save_training_state(basho_id, day, num_bouts, accuracy):
    """Save the training state"""
    state = {
        'last_trained_basho_id': basho_id,
        'last_trained_day': day,
        'last_training_date': datetime.now().isoformat(),
        'num_training_bouts': num_bouts,
        'accuracy': accuracy
    }

    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def get_latest_bout_in_db():
    """Query database for the most recent bout"""
    loader = SumoDataLoader()

    try:
        import pymysql
        connection = pymysql.connect(**loader.conn_params)
        cursor = connection.cursor()

        query = """
        SELECT basho_id, day
        FROM boi_ozumobout
        ORDER BY basho_id DESC, day DESC
        LIMIT 1
        """

        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        connection.close()

        if result:
            return result[0], result[1]  # basho_id, day
        return None, None

    except Exception as e:
        print(f"Error querying database: {e}")
        return None, None

def update_model(use_full_corpus=True, verbose=True):
    """
    Update the model with new bout data.

    Args:
        use_full_corpus: If True, train on ALL bouts (recommended). If False, only train on recent data.
        verbose: Show progress information

    Returns:
        dict with update information
    """
    if verbose:
        print("=" * 60)
        print("SUMO PREDICTION MODEL - INCREMENTAL UPDATE")
        print("=" * 60)
        print()

    # Load current training state
    state = load_training_state()

    if verbose:
        print("Current Training State:")
        if state['last_trained_basho_id']:
            print(f"  Last trained: Basho {state['last_trained_basho_id']}, Day {state['last_trained_day']}")
            print(f"  Training date: {state['last_training_date']}")
            print(f"  Training bouts: {state['num_training_bouts']:,}")
            print(f"  Accuracy: {state['accuracy']*100:.2f}%" if state['accuracy'] else "  Accuracy: N/A")
        else:
            print("  No previous training found - will train from scratch")
        print()

    # Check for new data
    latest_basho, latest_day = get_latest_bout_in_db()

    if latest_basho is None:
        print("ERROR: Could not connect to database or no bouts found")
        return None

    if verbose:
        print(f"Latest bout in database: Basho {latest_basho}, Day {latest_day}")

    # Check if update is needed
    if (state['last_trained_basho_id'] == latest_basho and
        state['last_trained_day'] == latest_day):
        if verbose:
            print("\n✓ Model is already up-to-date!")
            print("  No new bouts to train on.")
        return {
            'updated': False,
            'message': 'Model already up-to-date',
            'last_basho': latest_basho,
            'last_day': latest_day
        }

    if verbose:
        print(f"\nNew data available! Updating model...")
        if use_full_corpus:
            print(f"Training on FULL CORPUS (all bouts through basho {latest_basho})")
        else:
            print(f"Training on RECENT DATA ONLY")
        print()

    # Determine training range
    if use_full_corpus:
        # Train on all available data
        start_basho = None  # None means all data
        end_basho = latest_basho
    else:
        # Legacy mode: use rolling window
        lookback_bashos = 50
        start_basho = max(491, latest_basho - lookback_bashos)
        end_basho = latest_basho

    if verbose:
        if start_basho:
            print(f"Training on bashos {start_basho} to {end_basho}")
        else:
            print(f"Training on ALL bashos through {end_basho}")
        print()
        print("Starting training process...")
        print("-" * 60)

    # Train the model
    result = train_and_save_production_model(
        start_basho=start_basho,
        end_basho=end_basho,
        latest_basho=latest_basho,
        latest_day=latest_day,
        verbose=verbose
    )

    if result and 'accuracy' in result:
        # Save training state
        save_training_state(
            latest_basho,
            latest_day,
            result['num_training_bouts'],
            result['accuracy']
        )

        if verbose:
            print()
            print("=" * 60)
            print("✓ MODEL UPDATE COMPLETE")
            print("=" * 60)
            print(f"  New training bouts: {result['num_training_bouts']:,}")
            print(f"  New accuracy: {result['accuracy']*100:.2f}%")
            print(f"  Model saved to: models/sumo_predictor_production.joblib")
            print()

        return {
            'updated': True,
            'last_basho': latest_basho,
            'last_day': latest_day,
            'num_bouts': result['num_training_bouts'],
            'accuracy': result['accuracy']
        }
    else:
        if verbose:
            print("\n✗ Training failed")
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Update the sumo prediction model with new data')
    parser.add_argument('--recent-only', action='store_true',
                        help='Train on recent data only (not recommended, use for testing)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output messages')

    args = parser.parse_args()

    result = update_model(use_full_corpus=not args.recent_only, verbose=not args.quiet)

    if result and result['updated']:
        sys.exit(0)
    elif result and not result['updated']:
        sys.exit(0)  # Already up-to-date is success
    else:
        sys.exit(1)  # Error occurred
