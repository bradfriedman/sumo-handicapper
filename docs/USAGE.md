# Usage Guide - Sumo Handicapper

Detailed guide for making predictions using CLI tools and understanding outputs.

## Making Predictions

### Interactive Mode (Easiest!)

**By Name:**
```bash
uv run python predict_by_name.py --interactive
```

Prompts for:
- Basho ID (tournament)
- Day (1-15)
- Wrestler A name (supports partial matches!)
- Wrestler B name

**By ID:**
```bash
uv run python predict_bouts.py --interactive
```

Prompts for wrestler IDs instead of names.

### Single Predictions

**By name (recommended):**
```bash
uv run python predict_by_name.py \
  --name1 "Hakuho" \
  --name2 "Asashoryu" \
  --basho 630 \
  --day 10
```

**Partial names work:**
```bash
uv run python predict_by_name.py \
  --name1 "Haku" \
  --name2 "Asa" \
  --basho 630 \
  --day 10
# Will show you all wrestlers matching those partial names
```

**By ID:**
```bash
uv run python predict_bouts.py \
  --rikishi1 30 \
  --rikishi2 245 \
  --basho 630 \
  --day 10
```

### Batch Predictions

Create a CSV file with bout details:

```csv
rikishi_a_id,rikishi_b_id,basho_id,day,rikishi_a_rank,rikishi_b_rank
123,456,630,1,5,8
789,321,630,5,2,10
```

Run predictions:
```bash
uv run python predict_bouts.py --csv your_bouts.csv
```

Results saved to `your_bouts_predictions.csv`

## CSV Format

**Required columns:**
- `rikishi_a_id` - ID of first wrestler
- `rikishi_b_id` - ID of second wrestler
- `basho_id` - Tournament ID (491-630 in historical data)
- `day` - Day of tournament (1-15)

**Optional columns:**
- `rikishi_a_rank` - Rank number for wrestler A
- `rikishi_b_rank` - Rank number for wrestler B
- `rikishi_a_dob` - Date of birth for wrestler A
- `rikishi_b_dob` - Date of birth for wrestler B

See [data/sample_bouts.csv](../data/sample_bouts.csv) for example.

## Understanding Output

### Win Probabilities

```
Loading prediction model...
Rikishi 30 win probability: 25.4%
Rikishi 245 win probability: 74.6%

🎯 Predicted winner: Rikishi 245
   Confidence: 74.6%
```

- **50% = coin flip** (toss-up match)
- **70%+ = high confidence** (strong favorite)
- **90%+ = very likely** (heavy favorite)

### Head-to-Head Record

```
📊 Career Head-to-Head: 21-36
```

Shows historical matchup record between these two wrestlers.

### Fantasy League Points

**Scoring System:**
- **Base points:** 2 points for any win
- **Upset bonus:** Additional points when lower-ranked wrestler wins higher-ranked opponent

**Rank Groups:**
- Yokozuna (rank -3)
- Ozeki (rank -2)
- Sekiwake (rank -1)
- Komusubi (rank 0)
- M1-4 (ranks 1-4)
- M5-8 (ranks 5-8)
- M9-M12 (ranks 9-12)
- M13+ (ranks 13-20)

**Examples:**
- M13 defeats M5: 2 base + 2 upset = **4 points** (2 groups lower)
- Sekiwake defeats Komusubi: 2 base + 0 upset = **2 points** (higher rank wins)
- M12 defeats Yokozuna: 2 base + 6 upset = **8 points** (major upset!)

**Expected Points:**
Win probability × Potential points = Expected value

### Key Features

The prediction output shows important factors:

- **elo_diff:** Difference in Elo ratings (skill level)
- **rank_diff:** Difference in rankings
- **momentum_diff:** Recent form difference
- **experience_diff:** Experience gap
- **h2h_record:** Historical head-to-head performance

## Model Information

- **Accuracy:** ~60.4% on historical test data
- **Models:** Ensemble of Random Forest (45%), LightGBM (45%), XGBoost (10%)
- **Training Data:** 40,718 bouts from bashos 491-630
- **Features:** 41 features including Elo, rank, momentum, h2h, win rates, experience

## Notes

- Model requires wrestlers to have historical bout data
- Predictions work best for wrestlers who have fought recently
- Rank information (if available) improves accuracy
- ~60% accuracy reflects the inherent unpredictability of sports
- First-time matchups won't have head-to-head history
