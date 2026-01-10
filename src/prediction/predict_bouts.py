"""
Predict sumo bout outcomes using the trained model

Usage:
  python3 src/prediction/predict_bouts.py --csv bouts.csv       # Predict from CSV file
  python3 src/prediction/predict_bouts.py --interactive         # Interactive manual entry
  python3 src/prediction/predict_bouts.py --rikishi1 ID1 --rikishi2 ID2 --basho BASHO_ID --day DAY  # Single prediction
"""
import pandas as pd
import argparse
import sys
from src.prediction.prediction_engine import load_model, predict_bout
from src.core.fantasy_points import get_rank_label

def predict_from_csv(model_package, csv_file):
    """Predict outcomes for bouts in a CSV file"""
    print(f"Loading bouts from {csv_file}...")

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Error: File '{csv_file}' not found")
        return
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Validate required columns
    required = ['rikishi_a_id', 'rikishi_b_id', 'basho_id', 'day']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ Error: CSV missing required columns: {missing}")
        print(f"   Required columns: {required}")
        print(f"   Optional columns: rikishi_a_rank, rikishi_b_rank, rikishi_a_dob, rikishi_b_dob")
        return

    print(f"Found {len(df)} bouts to predict\n")
    print("="*80)

    results = []
    for idx, row in df.iterrows():
        result = predict_bout(
            model_package,
            row['rikishi_a_id'],
            row['rikishi_b_id'],
            row['basho_id'],
            row['day'],
            row.get('rikishi_a_rank'),
            row.get('rikishi_b_rank'),
            row.get('rikishi_a_dob'),
            row.get('rikishi_b_dob')
        )

        if 'error' in result:
            print(f"Bout {idx+1}: ❌ {result['error']}")
        else:
            winner_id = result['predicted_winner_id']
            conf = result['confidence'] * 100
            print(f"Bout {idx+1}: Rikishi {row['rikishi_a_id']} vs {row['rikishi_b_id']}")
            print(f"  → Predicted winner: Rikishi {winner_id} ({conf:.1f}% confidence)")
            print(f"  → Win probabilities: A={result['rikishi_a_win_probability']:.1%}, "
                  f"B={result['rikishi_b_win_probability']:.1%}")

        results.append(result)
        print()

    # Save results
    output_file = csv_file.replace('.csv', '_predictions.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print("="*80)
    print(f"✅ Predictions saved to: {output_file}")

def interactive_mode(model_package):
    """Interactive mode for manual entry"""
    print("="*80)
    print("INTERACTIVE PREDICTION MODE")
    print("="*80)
    print("\nEnter bout details (or 'quit' to exit):\n")

    while True:
        try:
            # Get Rikishi A ID
            rikishi_a_id = None
            while rikishi_a_id is None:
                rikishi_a = input("Rikishi A ID: ").strip()
                if rikishi_a.lower() == 'quit':
                    return
                try:
                    rikishi_a_id = int(rikishi_a)
                except ValueError:
                    print("❌ Invalid input. Please enter a valid integer for Rikishi A ID.")

            # Get Rikishi B ID
            rikishi_b_id = None
            while rikishi_b_id is None:
                rikishi_b = input("Rikishi B ID: ").strip()
                if rikishi_b.lower() == 'quit':
                    return
                try:
                    rikishi_b_id = int(rikishi_b)
                except ValueError:
                    print("❌ Invalid input. Please enter a valid integer for Rikishi B ID.")

            # Get Basho ID
            basho_id = None
            while basho_id is None:
                basho = input("Basho ID: ").strip()
                if basho.lower() == 'quit':
                    return
                try:
                    basho_id = int(basho)
                except ValueError:
                    print("❌ Invalid input. Please enter a valid integer for Basho ID.")

            # Get Day
            day_num = None
            while day_num is None:
                day = input("Day (1-15): ").strip()
                if day.lower() == 'quit':
                    return
                try:
                    day_num = int(day)
                    if not (1 <= day_num <= 15):
                        print("❌ Day must be between 1 and 15.")
                        day_num = None
                except ValueError:
                    print("❌ Invalid input. Please enter a valid integer for Day (1-15).")

            # Optional inputs
            rank_a = input("Rikishi A rank (optional, press Enter to skip): ").strip()
            rank_b = input("Rikishi B rank (optional, press Enter to skip): ").strip()

            # Convert optional inputs
            rank_a_val = None
            if rank_a:
                try:
                    rank_a_val = int(rank_a)
                except ValueError:
                    print("❌ Invalid rank for Rikishi A. Continuing without rank.")

            rank_b_val = None
            if rank_b:
                try:
                    rank_b_val = int(rank_b)
                except ValueError:
                    print("❌ Invalid rank for Rikishi B. Continuing without rank.")

            print("\nPredicting...")
            result = predict_bout(
                model_package,
                rikishi_a_id,
                rikishi_b_id,
                basho_id,
                day_num,
                rank_a_val,
                rank_b_val
            )

            print("\n" + "="*80)
            if 'error' in result:
                print(f"❌ {result['error']}")
                if 'suggestion' in result:
                    print(f"   {result['suggestion']}")
            else:
                print("PREDICTION RESULTS")
                print("="*80)
                print(f"\nRikishi A (ID {rikishi_a_id}) win probability: {result['rikishi_a_win_probability']:.1%}")
                print(f"Rikishi B (ID {rikishi_b_id}) win probability: {result['rikishi_b_win_probability']:.1%}")
                print(f"\n🎯 Predicted winner: Rikishi {result['predicted_winner_id']}")
                print(f"   Confidence: {result['confidence']:.1%}")

                # Head-to-head record
                if 'head_to_head' in result:
                    h2h = result['head_to_head']
                    h2h_total = h2h['rikishi_a_wins'] + h2h['rikishi_b_wins']
                    if h2h_total > 0:
                        print(f"\n📊 Career Head-to-Head: {h2h['rikishi_a_wins']}-{h2h['rikishi_b_wins']} (Rikishi A leads)")
                    else:
                        print(f"\n📊 Career Head-to-Head: 0-0 (first meeting)")

                # Fantasy points
                fp = result['fantasy_points']
                if fp['rikishi_a_expected'] is not None:
                    print("\n🎮 Fantasy League Points:")
                    rank_a_label = get_rank_label(fp['rikishi_a_rank'])
                    rank_b_label = get_rank_label(fp['rikishi_b_rank'])
                    print(f"  Rikishi A ({rank_a_label}): {fp['rikishi_a_expected']:.2f} expected pts (max {fp['rikishi_a_potential']} if win)")
                    print(f"  Rikishi B ({rank_b_label}): {fp['rikishi_b_expected']:.2f} expected pts (max {fp['rikishi_b_potential']} if win)")

                print("\nKey factors:")
                for key, value in result['key_features'].items():
                    if value != 'N/A':
                        print(f"  {key}: {value:.3f}")

                print("\nIndividual model predictions (A win probability):")
                for model, prob in result['individual_predictions'].items():
                    print(f"  {model}: {prob:.1%}")

            print("="*80)
            print()

        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            print()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

def main():
    parser = argparse.ArgumentParser(
        description='Predict sumo bout outcomes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python3 predict_bouts.py --interactive

  # Predict from CSV file
  python3 predict_bouts.py --csv my_bouts.csv

  # Single bout prediction
  python3 predict_bouts.py --rikishi1 123 --rikishi2 456 --basho 630 --day 10

CSV Format:
  Required columns: rikishi_a_id, rikishi_b_id, basho_id, day
  Optional columns: rikishi_a_rank, rikishi_b_rank, rikishi_a_dob, rikishi_b_dob
        """
    )

    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode for manual entry')
    parser.add_argument('--csv', type=str,
                        help='CSV file with bouts to predict')
    parser.add_argument('--rikishi1', type=int,
                        help='Rikishi A ID (for single prediction)')
    parser.add_argument('--rikishi2', type=int,
                        help='Rikishi B ID (for single prediction)')
    parser.add_argument('--basho', type=int,
                        help='Basho ID (for single prediction)')
    parser.add_argument('--day', type=int,
                        help='Day number 1-15 (for single prediction)')

    args = parser.parse_args()

    # Load model
    print("\nLoading prediction model...")
    model_package = load_model()

    # Determine mode
    if args.interactive:
        interactive_mode(model_package)
    elif args.csv:
        predict_from_csv(model_package, args.csv)
    elif all([args.rikishi1, args.rikishi2, args.basho, args.day]):
        result = predict_bout(
            model_package,
            args.rikishi1,
            args.rikishi2,
            args.basho,
            args.day
        )

        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            print("="*80)
            print("PREDICTION RESULTS")
            print("="*80)
            print(f"\nRikishi {args.rikishi1} win probability: {result['rikishi_a_win_probability']:.1%}")
            print(f"Rikishi {args.rikishi2} win probability: {result['rikishi_b_win_probability']:.1%}")
            print(f"\n🎯 Predicted winner: Rikishi {result['predicted_winner_id']}")
            print(f"   Confidence: {result['confidence']:.1%}")

            # Head-to-head record
            if 'head_to_head' in result:
                h2h = result['head_to_head']
                h2h_total = h2h['rikishi_a_wins'] + h2h['rikishi_b_wins']
                if h2h_total > 0:
                    print(f"\n📊 Career Head-to-Head: {h2h['rikishi_a_wins']}-{h2h['rikishi_b_wins']} (Rikishi {args.rikishi1} vs {args.rikishi2})")
                else:
                    print(f"\n📊 Career Head-to-Head: 0-0 (first meeting)")

            # Fantasy points
            fp = result['fantasy_points']
            if fp['rikishi_a_expected'] is not None:
                print("\n🎮 Fantasy League Points:")
                rank_a_label = get_rank_label(fp['rikishi_a_rank'])
                rank_b_label = get_rank_label(fp['rikishi_b_rank'])
                print(f"  Rikishi {args.rikishi1} ({rank_a_label}): {fp['rikishi_a_expected']:.2f} expected pts (max {fp['rikishi_a_potential']} if win)")
                print(f"  Rikishi {args.rikishi2} ({rank_b_label}): {fp['rikishi_b_expected']:.2f} expected pts (max {fp['rikishi_b_potential']} if win)")
            print("="*80)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
