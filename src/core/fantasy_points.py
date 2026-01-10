"""
Fantasy League Points Calculator

Scoring system:
- Base points: 2 points for a win, 0 for a loss
- Upset points: Based on group difference (lower rank defeating higher rank)
- Expected points: potential_points * win_probability

Rank Groups:
- Yokozuna (rank -3): group level -3
- Ozeki (rank -2): group level -2
- Sekiwake (rank -1): group level -1
- Komusubi (rank 0): group level 0
- M1-4 (ranks 1-4): group level 1
- M5-8 (ranks 5-8): group level 2
- M9-M12 (ranks 9-12): group level 3
- M13+ (ranks 13-20): group level 4

Example:
- M13 defeats M5: 2 base + 2 upset (group 4 - group 2) = 4 points
- Sekiwake defeats Komusubi: 2 base + 0 upset (no upset when higher rank wins) = 2 points
"""

def get_group_level(rank):
    """
    Get the group level for a given rank

    Args:
        rank: Integer rank from database (-3 for Yokozuna, positive for Maegashira)

    Returns:
        Group level integer, or None if rank is invalid
    """
    if rank is None:
        return None

    if rank == -3:  # Yokozuna
        return -3
    elif rank == -2:  # Ozeki
        return -2
    elif rank == -1:  # Sekiwake
        return -1
    elif rank == 0:  # Komusubi
        return 0
    elif 1 <= rank <= 4:  # M1-4
        return 1
    elif 5 <= rank <= 8:  # M5-8
        return 2
    elif 9 <= rank <= 12:  # M9-M12
        return 3
    elif rank >= 13:  # M13+ (including Juryo which we ignore)
        return 4
    else:
        return None


def get_rank_label(rank):
    """Get a human-readable label for a rank"""
    if rank is None:
        return "Unknown"
    elif rank == -3:
        return "Yokozuna"
    elif rank == -2:
        return "Ozeki"
    elif rank == -1:
        return "Sekiwake"
    elif rank == 0:
        return "Komusubi"
    elif rank >= 1:
        return f"M{rank}"
    else:
        return f"Rank {rank}"


def calculate_fantasy_points(winner_rank, loser_rank):
    """
    Calculate fantasy points for a winner given their rank and opponent's rank

    Args:
        winner_rank: Rank of the winning rikishi
        loser_rank: Rank of the losing rikishi

    Returns:
        Total fantasy points (base + upset), or None if ranks are invalid
    """
    winner_group = get_group_level(winner_rank)
    loser_group = get_group_level(loser_rank)

    if winner_group is None or loser_group is None:
        return None

    # Base points for any win
    base_points = 2

    # Upset points: only if winner is lower ranked (higher group level)
    upset_points = max(0, winner_group - loser_group)

    return base_points + upset_points


def calculate_expected_points(rank_a, rank_b, win_prob_a, win_prob_b):
    """
    Calculate expected fantasy points for both rikishi

    Args:
        rank_a: Rank of rikishi A
        rank_b: Rank of rikishi B
        win_prob_a: Probability that rikishi A wins (0-1)
        win_prob_b: Probability that rikishi B wins (0-1)

    Returns:
        Tuple of (expected_points_a, expected_points_b, potential_points_a, potential_points_b)
        Returns None values if ranks are invalid
    """
    # Calculate potential points if each rikishi wins
    potential_points_a = calculate_fantasy_points(rank_a, rank_b)
    potential_points_b = calculate_fantasy_points(rank_b, rank_a)

    if potential_points_a is None or potential_points_b is None:
        return None, None, None, None

    # Calculate expected value
    expected_points_a = potential_points_a * win_prob_a
    expected_points_b = potential_points_b * win_prob_b

    return expected_points_a, expected_points_b, potential_points_a, potential_points_b


# Example usage and tests
if __name__ == "__main__":
    print("="*80)
    print("FANTASY POINTS CALCULATOR - TEST CASES")
    print("="*80)

    # Test case 1: M13 defeats M5 (example from user)
    print("\nTest 1: M13 defeats M5")
    points = calculate_fantasy_points(winner_rank=13, loser_rank=5)
    print(f"  Expected: 4 points (2 base + 2 upset)")
    print(f"  Actual:   {points} points")
    assert points == 4, f"Test 1 failed: expected 4, got {points}"

    # Test case 2: Sekiwake defeats Komusubi (example from user)
    print("\nTest 2: Sekiwake defeats Komusubi")
    points = calculate_fantasy_points(winner_rank=-1, loser_rank=0)
    print(f"  Expected: 2 points (2 base + 0 upset)")
    print(f"  Actual:   {points} points")
    assert points == 2, f"Test 2 failed: expected 2, got {points}"

    # Test case 3: Expected value calculation
    print("\nTest 3: Expected points - M1 vs Komusubi (50/50)")
    exp_a, exp_b, pot_a, pot_b = calculate_expected_points(
        rank_a=1, rank_b=0, win_prob_a=0.5, win_prob_b=0.5
    )
    print(f"  M1 potential if win: {pot_a} points")
    print(f"  M1 expected (50%):   {exp_a:.2f} points")
    print(f"  Komusubi potential:  {pot_b} points")
    print(f"  Komusubi expected:   {exp_b:.2f} points")

    # Test case 4: Yokozuna vs M1
    print("\nTest 4: Yokozuna vs M1")
    exp_a, exp_b, pot_a, pot_b = calculate_expected_points(
        rank_a=-3, rank_b=1, win_prob_a=0.8, win_prob_b=0.2
    )
    print(f"  Yokozuna potential:  {pot_a} points (no upset for higher rank)")
    print(f"  Yokozuna expected:   {exp_a:.2f} points")
    print(f"  M1 potential:        {pot_b} points (major upset!)")
    print(f"  M1 expected:         {exp_b:.2f} points")

    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)
