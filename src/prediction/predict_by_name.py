"""
Predict sumo bout outcomes using rikishi names instead of IDs

Usage:
  python3 -m src.prediction.predict_by_name --interactive         # Interactive mode with name lookup
  python3 -m src.prediction.predict_by_name --name1 "Hakuho" --name2 "Asashoryu" --basho 630 --day 5
"""
import pandas as pd
import pymysql
import argparse
import sys
from src.prediction.prediction_engine import load_model, predict_bout, search_rikishi_by_name, DB_CONFIG
from src.core.fantasy_points import calculate_expected_points, get_rank_label

def select_rikishi(name_query, basho_id, rikishi_label="Rikishi"):
    """
    Search for rikishi and let user select if multiple matches
    Returns: (rikishi_id, rank, dob) or None if cancelled
    """
    results = search_rikishi_by_name(name_query, basho_id)

    if not results:
        print(f"❌ No rikishi found matching '{name_query}'")
        print("   Try a different spelling or partial name")
        return None

    if len(results) == 1:
        rikishi = results[0]
        # Prefer showing ring name (shikona) if available
        display_name = rikishi.get('ring_name') or rikishi['real_name']
        name_detail = f" (real name: {rikishi['real_name']})" if rikishi.get('ring_name') else ""
        rank_info = f" (Rank: {rikishi['rank']})" if rikishi.get('rank') else ""
        print(f"✓ Found: {display_name}{name_detail}{rank_info}")
        return (rikishi['id'], rikishi.get('rank'), rikishi.get('dob'))

    # Multiple matches - let user choose
    print(f"\nMultiple rikishi found matching '{name_query}':")
    for idx, rikishi in enumerate(results, 1):
        # Prefer showing ring name (shikona) if available
        display_name = rikishi.get('ring_name') or rikishi['real_name']
        name_detail = f" / {rikishi['real_name']}" if rikishi.get('ring_name') else ""
        rank_info = f" (Rank: {rikishi['rank']})" if rikishi.get('rank') else " (Not in this basho)"
        print(f"  [{idx}] {display_name}{name_detail}{rank_info}")

    while True:
        try:
            choice = input(f"\nSelect {rikishi_label} [1-{len(results)}] or 'q'=cancel: ").strip()
            if choice.lower() in ['q', 'quit']:
                return None

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(results):
                selected = results[choice_idx]
                return (selected['id'], selected.get('rank'), selected.get('dob'))
            else:
                print(f"Please enter a number between 1 and {len(results)}")
        except ValueError:
            print("Please enter a valid number")

def predict_by_names(model_package, name1, name2, basho_id, day):
    """
    Predict bout outcome using rikishi names
    """
    print(f"\nBasho ID: {basho_id}, Day: {day}")
    print("="*80)

    # Look up first rikishi
    print(f"\nSearching for '{name1}'...")
    rikishi1_search = search_rikishi_by_name(name1, basho_id)
    if not rikishi1_search:
        print(f"❌ No rikishi found matching '{name1}'")
        return None

    # Get the rikishi info
    if len(rikishi1_search) == 1:
        rikishi1 = rikishi1_search[0]
        display_name1 = rikishi1.get('ring_name') or rikishi1['real_name']
        name_detail1 = f" (real name: {rikishi1['real_name']})" if rikishi1.get('ring_name') else ""
        rank_info1 = f" (Rank: {rikishi1['rank']})" if rikishi1.get('rank') else ""
        print(f"✓ Found: {display_name1}{name_detail1}{rank_info1}")
        rikishi1_id = rikishi1['id']
        rank1 = rikishi1.get('rank')
        dob1 = rikishi1.get('dob')
    else:
        # Multiple matches - let user choose
        print(f"\nMultiple rikishi found matching '{name1}':")
        for idx, rikishi in enumerate(rikishi1_search, 1):
            display_name = rikishi.get('ring_name') or rikishi['real_name']
            name_detail = f" / {rikishi['real_name']}" if rikishi.get('ring_name') else ""
            rank_info = f" (Rank: {rikishi['rank']})" if rikishi.get('rank') else " (Not in this basho)"
            print(f"  [{idx}] {display_name}{name_detail}{rank_info}")

        while True:
            try:
                choice = input(f"\nSelect Rikishi A [1-{len(rikishi1_search)}] or 'q'=cancel: ").strip()
                if choice.lower() in ['q', 'quit']:
                    return None
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(rikishi1_search):
                    rikishi1 = rikishi1_search[choice_idx]
                    display_name1 = rikishi1.get('ring_name') or rikishi1['real_name']
                    rikishi1_id = rikishi1['id']
                    rank1 = rikishi1.get('rank')
                    dob1 = rikishi1.get('dob')
                    break
                else:
                    print(f"Please enter a number between 1 and {len(rikishi1_search)}")
            except ValueError:
                print("Please enter a valid number")

    # Look up second rikishi
    print(f"\nSearching for '{name2}'...")
    rikishi2_search = search_rikishi_by_name(name2, basho_id)
    if not rikishi2_search:
        print(f"❌ No rikishi found matching '{name2}'")
        return None

    if len(rikishi2_search) == 1:
        rikishi2 = rikishi2_search[0]
        display_name2 = rikishi2.get('ring_name') or rikishi2['real_name']
        name_detail2 = f" (real name: {rikishi2['real_name']})" if rikishi2.get('ring_name') else ""
        rank_info2 = f" (Rank: {rikishi2['rank']})" if rikishi2.get('rank') else ""
        print(f"✓ Found: {display_name2}{name_detail2}{rank_info2}")
        rikishi2_id = rikishi2['id']
        rank2 = rikishi2.get('rank')
        dob2 = rikishi2.get('dob')
    else:
        # Multiple matches - let user choose
        print(f"\nMultiple rikishi found matching '{name2}':")
        for idx, rikishi in enumerate(rikishi2_search, 1):
            display_name = rikishi.get('ring_name') or rikishi['real_name']
            name_detail = f" / {rikishi['real_name']}" if rikishi.get('ring_name') else ""
            rank_info = f" (Rank: {rikishi['rank']})" if rikishi.get('rank') else " (Not in this basho)"
            print(f"  [{idx}] {display_name}{name_detail}{rank_info}")

        while True:
            try:
                choice = input(f"\nSelect Rikishi B [1-{len(rikishi2_search)}] or 'q'=cancel: ").strip()
                if choice.lower() in ['q', 'quit']:
                    return None
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(rikishi2_search):
                    rikishi2 = rikishi2_search[choice_idx]
                    display_name2 = rikishi2.get('ring_name') or rikishi2['real_name']
                    rikishi2_id = rikishi2['id']
                    rank2 = rikishi2.get('rank')
                    dob2 = rikishi2.get('dob')
                    break
                else:
                    print(f"Please enter a number between 1 and {len(rikishi2_search)}")
            except ValueError:
                print("Please enter a valid number")

    # Make prediction
    print("\n" + "="*80)
    print("Making prediction...")
    print("="*80)

    result = predict_bout(
        model_package,
        rikishi1_id,
        rikishi2_id,
        basho_id,
        day,
        rank1,
        rank2,
        dob1,
        dob2
    )

    # Return result with names for display
    return result, {
        'rikishi1_id': rikishi1_id,
        'rikishi1_name': display_name1,
        'rikishi2_id': rikishi2_id,
        'rikishi2_name': display_name2
    }

def interactive_mode_with_names(model_package):
    """Interactive mode using rikishi names"""
    print("="*80)
    print("INTERACTIVE PREDICTION MODE (Name-based)")
    print("="*80)
    print("\nEnter bout details using rikishi names")
    print("Commands: 'q'=quit, 'c'=change basho/day")
    print("Tip: You can use partial names (e.g., 'Haku' for 'Hakuho')\n")

    # Store basho and day across predictions
    basho_id = None
    day = None

    while True:
        try:
            # Ask for basho and day if not set, or if user wants to change
            if basho_id is None:
                # Keep asking for valid basho_id
                while basho_id is None:
                    basho_input = input("\nBasho ID: ").strip()
                    if basho_input.lower() in ['quit', 'q']:
                        return  # Exit the function entirely

                    try:
                        basho_id = int(basho_input)
                    except ValueError:
                        print("❌ Invalid input. Please enter a valid integer for Basho ID.")
                        continue

                # Keep asking for valid day
                while day is None:
                    day_input = input("Day (1-15): ").strip()
                    if day_input.lower() in ['quit', 'q']:
                        return  # Exit the function entirely

                    try:
                        day = int(day_input)
                        if not (1 <= day <= 15):
                            print("❌ Day must be between 1 and 15.")
                            day = None
                            continue
                    except ValueError:
                        print("❌ Invalid input. Please enter a valid integer for Day (1-15).")
                        continue

                print(f"\n✓ Using Basho {basho_id}, Day {day} for predictions")
                print("  (Type 'c' or 'change' to change basho/day)\n")
            else:
                # Show current basho/day
                print(f"\nCurrent: Basho {basho_id}, Day {day}")

            # Get rikishi names
            name1 = input("Rikishi A name (or 'c'=change basho/day, 'q'=quit): ").strip()
            if name1.lower() in ['quit', 'q']:
                break
            elif name1.lower() in ['change', 'c']:
                basho_id = None
                day = None
                continue

            name2 = input("Rikishi B name (or partial name): ").strip()
            if name2.lower() in ['quit', 'q']:
                break

            # Make prediction
            result_tuple = predict_by_names(model_package, name1, name2, basho_id, day)

            if result_tuple:
                result, names_dict = result_tuple

                if 'error' in result:
                    print(f"\n❌ {result['error']}")
                    if 'suggestion' in result:
                        print(f"   {result['suggestion']}")
                else:
                    print(f"\n{names_dict['rikishi1_name']} win probability: {result['rikishi_a_win_probability']:.1%}")
                    print(f"{names_dict['rikishi2_name']} win probability: {result['rikishi_b_win_probability']:.1%}")

                    # Determine winner name
                    winner_name = names_dict['rikishi1_name'] if result['predicted_winner_id'] == names_dict['rikishi1_id'] else names_dict['rikishi2_name']
                    print(f"\n🎯 Predicted winner: {winner_name}")
                    print(f"   Confidence: {result['confidence']:.1%}")

                    # Head-to-head record
                    if 'head_to_head' in result:
                        h2h = result['head_to_head']
                        h2h_total = h2h['rikishi_a_wins'] + h2h['rikishi_b_wins']
                        if h2h_total > 0:
                            print(f"\n📊 Career Head-to-Head: {h2h['rikishi_a_wins']}-{h2h['rikishi_b_wins']} ({names_dict['rikishi1_name']} vs {names_dict['rikishi2_name']})")
                        else:
                            print(f"\n📊 Career Head-to-Head: 0-0 (first meeting)")

                    # Fantasy points
                    fp = result['fantasy_points']
                    if fp['rikishi_a_expected'] is not None:
                        print("\n🎮 Fantasy League Points:")
                        rank_a_label = get_rank_label(fp['rikishi_a_rank'])
                        rank_b_label = get_rank_label(fp['rikishi_b_rank'])
                        print(f"  {names_dict['rikishi1_name']} ({rank_a_label}): {fp['rikishi_a_expected']:.2f} expected pts (max {fp['rikishi_a_potential']} if win)")
                        print(f"  {names_dict['rikishi2_name']} ({rank_b_label}): {fp['rikishi_b_expected']:.2f} expected pts (max {fp['rikishi_b_potential']} if win)")

                    print("\nKey factors:")
                    for key, value in result['key_features'].items():
                        if value != 'N/A':
                            print(f"  {key}: {value:.3f}")

                    print("\nIndividual model predictions ({} win probability):".format(names_dict['rikishi1_name']))
                    for model, prob in result['individual_predictions'].items():
                        print(f"  {model}: {prob:.1%}")

            print("\n" + "="*80)

        except ValueError as e:
            print(f"❌ Invalid input: {e}")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description='Predict sumo bout outcomes using rikishi names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with name lookup
  python3 predict_by_name.py --interactive

  # Single prediction by names
  python3 predict_by_name.py --name1 "Hakuho" --name2 "Asashoryu" --basho 630 --day 5

  # Partial names work too
  python3 predict_by_name.py --name1 "Haku" --name2 "Asa" --basho 630 --day 5
        """
    )

    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode with name-based lookup')
    parser.add_argument('--name1', type=str,
                        help='Rikishi A name (or partial name)')
    parser.add_argument('--name2', type=str,
                        help='Rikishi B name (or partial name)')
    parser.add_argument('--basho', type=int,
                        help='Basho ID')
    parser.add_argument('--day', type=int,
                        help='Day number 1-15')

    args = parser.parse_args()

    # Load model
    print("\nLoading prediction model...")
    model_package = load_model()

    # Determine mode
    if args.interactive:
        interactive_mode_with_names(model_package)
    elif all([args.name1, args.name2, args.basho, args.day]):
        result_tuple = predict_by_names(
            model_package,
            args.name1,
            args.name2,
            args.basho,
            args.day
        )

        if result_tuple:
            result, names_dict = result_tuple

            if 'error' in result:
                print(f"\n❌ {result['error']}")
            else:
                print(f"\n{names_dict['rikishi1_name']} win probability: {result['rikishi_a_win_probability']:.1%}")
                print(f"{names_dict['rikishi2_name']} win probability: {result['rikishi_b_win_probability']:.1%}")

                # Determine winner name
                winner_name = names_dict['rikishi1_name'] if result['predicted_winner_id'] == names_dict['rikishi1_id'] else names_dict['rikishi2_name']
                print(f"\n🎯 Predicted winner: {winner_name}")
                print(f"   Confidence: {result['confidence']:.1%}")

                # Head-to-head record
                if 'head_to_head' in result:
                    h2h = result['head_to_head']
                    h2h_total = h2h['rikishi_a_wins'] + h2h['rikishi_b_wins']
                    if h2h_total > 0:
                        print(f"\n📊 Career Head-to-Head: {h2h['rikishi_a_wins']}-{h2h['rikishi_b_wins']} ({names_dict['rikishi1_name']} vs {names_dict['rikishi2_name']})")
                    else:
                        print(f"\n📊 Career Head-to-Head: 0-0 (first meeting)")

                # Fantasy points
                fp = result['fantasy_points']
                if fp['rikishi_a_expected'] is not None:
                    print("\n🎮 Fantasy League Points:")
                    rank_a_label = get_rank_label(fp['rikishi_a_rank'])
                    rank_b_label = get_rank_label(fp['rikishi_b_rank'])
                    print(f"  {names_dict['rikishi1_name']} ({rank_a_label}): {fp['rikishi_a_expected']:.2f} expected pts (max {fp['rikishi_a_potential']} if win)")
                    print(f"  {names_dict['rikishi2_name']} ({rank_b_label}): {fp['rikishi_b_expected']:.2f} expected pts (max {fp['rikishi_b_potential']} if win)")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
