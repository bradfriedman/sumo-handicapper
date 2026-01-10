"""
Iterative Model Improvement Script
Systematically tests different hyperparameters to find the best model
"""
import json
from datetime import datetime
from sumo_predictor import (
    ModelConfig, SumoDataLoader, FeatureEngineer, SumoPredictor
)
from sklearn.model_selection import train_test_split
import pandas as pd


def test_configuration(config: ModelConfig, iteration: int):
    """Test a single configuration and return results"""
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Elo K-factor: {config.elo_k_factor}")
    print(f"  Elo Initial Rating: {config.elo_initial_rating}")
    print(f"  Recent Bouts Window: {config.recent_bouts_window}")
    print(f"  Include Head-to-Head: {config.include_head_to_head}")
    print(f"  Include Age: {config.include_age}")

    try:
        # Load data
        loader = SumoDataLoader()
        bouts_df = loader.load_raw_bouts()

        # Build features
        engineer = FeatureEngineer(config)
        X, y = engineer.build_dataset(bouts_df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y
        )

        # Train models
        predictor = SumoPredictor(config)
        predictor.train_models(X_train, y_train)

        # Evaluate
        results = predictor.evaluate_models(X_test, y_test)

        # Extract metrics
        iteration_results = {
            'iteration': iteration,
            'config': {
                'elo_k_factor': config.elo_k_factor,
                'elo_initial_rating': config.elo_initial_rating,
                'recent_bouts_window': config.recent_bouts_window,
                'include_head_to_head': config.include_head_to_head,
                'include_age': config.include_age,
            },
            'models': {}
        }

        for model_name, result in results.items():
            iteration_results['models'][model_name] = {
                'accuracy': float(result['accuracy']),
                'roc_auc': float(result['roc_auc'])
            }
            print(f"\n{model_name.upper()}: Accuracy={result['accuracy']:.4f}, ROC-AUC={result['roc_auc']:.4f}")

        return iteration_results, predictor, X_test, y_test

    except Exception as e:
        print(f"\n❌ Configuration failed: {e}")
        return None, None, None, None


def main():
    """Run iterative improvement experiments"""
    print("="*80)
    print("SUMO MODEL ITERATIVE IMPROVEMENT")
    print("="*80)

    all_results = []

    # Baseline (current best)
    configs_to_test = [
        # Baseline
        {
            'name': 'Baseline',
            'elo_k_factor': 32,
            'elo_initial_rating': 1500,
            'recent_bouts_window': 10,
            'include_head_to_head': True,
            'include_age': True
        },
        # Vary Elo K-factor (most important feature)
        {
            'name': 'Lower K-factor (24)',
            'elo_k_factor': 24,
            'elo_initial_rating': 1500,
            'recent_bouts_window': 10,
            'include_head_to_head': True,
            'include_age': True
        },
        {
            'name': 'Higher K-factor (40)',
            'elo_k_factor': 40,
            'elo_initial_rating': 1500,
            'recent_bouts_window': 10,
            'include_head_to_head': True,
            'include_age': True
        },
        {
            'name': 'Much Higher K-factor (50)',
            'elo_k_factor': 50,
            'elo_initial_rating': 1500,
            'recent_bouts_window': 10,
            'include_head_to_head': True,
            'include_age': True
        },
        # Vary Recent Bouts Window
        {
            'name': 'Smaller Recent Window (5)',
            'elo_k_factor': 32,
            'elo_initial_rating': 1500,
            'recent_bouts_window': 5,
            'include_head_to_head': True,
            'include_age': True
        },
        {
            'name': 'Larger Recent Window (15)',
            'elo_k_factor': 32,
            'elo_initial_rating': 1500,
            'recent_bouts_window': 15,
            'include_head_to_head': True,
            'include_age': True
        },
        # Test without certain features
        {
            'name': 'Without Head-to-Head',
            'elo_k_factor': 32,
            'elo_initial_rating': 1500,
            'recent_bouts_window': 10,
            'include_head_to_head': False,
            'include_age': True
        },
        {
            'name': 'Without Age',
            'elo_k_factor': 32,
            'elo_initial_rating': 1500,
            'recent_bouts_window': 10,
            'include_head_to_head': True,
            'include_age': False
        },
        # Best combinations from above
        {
            'name': 'Best K-factor + Large Window',
            'elo_k_factor': 40,  # Will adjust based on results
            'elo_initial_rating': 1500,
            'recent_bouts_window': 15,
            'include_head_to_head': True,
            'include_age': True
        },
    ]

    best_model = None
    best_accuracy = 0
    best_config = None
    best_iteration = 0

    for i, config_dict in enumerate(configs_to_test, 1):
        print(f"\n{'#'*80}")
        print(f"Testing: {config_dict['name']}")
        print(f"{'#'*80}")

        # Create config
        config = ModelConfig(
            elo_k_factor=config_dict['elo_k_factor'],
            elo_initial_rating=config_dict['elo_initial_rating'],
            recent_bouts_window=config_dict['recent_bouts_window'],
            include_head_to_head=config_dict['include_head_to_head'],
            include_age=config_dict['include_age']
        )

        # Test configuration
        results, predictor, X_test, y_test = test_configuration(config, i)

        if results:
            results['name'] = config_dict['name']
            all_results.append(results)

            # Track best model (using Random Forest as it's always available)
            if 'random_forest' in results['models']:
                accuracy = results['models']['random_forest']['accuracy']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = predictor
                    best_config = config_dict
                    best_iteration = i
                    print(f"\n🎯 NEW BEST MODEL! Accuracy: {accuracy:.4f}")

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Create results DataFrame
    summary_data = []
    for result in all_results:
        row = {
            'Iteration': result['iteration'],
            'Name': result['name'],
            'Elo K': result['config']['elo_k_factor'],
            'Recent Window': result['config']['recent_bouts_window'],
        }
        for model_name, metrics in result['models'].items():
            row[f'{model_name}_acc'] = metrics['accuracy']
            row[f'{model_name}_auc'] = metrics['roc_auc']
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"improvement_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Best model
    print(f"\n{'='*80}")
    print(f"BEST MODEL")
    print(f"{'='*80}")
    print(f"Iteration: {best_iteration}")
    print(f"Configuration: {best_config['name']}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"\nConfig details:")
    for key, value in best_config.items():
        if key != 'name':
            print(f"  {key}: {value}")

    # Save best model
    if best_model:
        best_model.save_models(f'sumo_models_best_{timestamp}.joblib')
        print(f"\nBest model saved to: sumo_models_best_{timestamp}.joblib")

    return all_results, best_model, best_config


if __name__ == '__main__':
    results, best_model, best_config = main()
