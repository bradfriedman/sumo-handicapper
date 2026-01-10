# Sumo Bout Prediction System

This system uses a trained ensemble model (60.4% accuracy) to predict the outcomes of sumo bouts.

## Quick Start

### 1a. Interactive Mode with Names (EASIEST!)

```bash
python3 predict_by_name.py --interactive
```

This will prompt you to enter rikishi **names** instead of IDs:
- Basho ID
- Day (1-15)
- Rikishi A name (supports partial matches!)
- Rikishi B name (supports partial matches!)

### 1b. Interactive Mode with IDs

```bash
python3 predict_bouts.py --interactive
```

This will prompt you to enter bout details using rikishi IDs:
- Rikishi A ID
- Rikishi B ID
- Basho ID
- Day (1-15)
- Ranks (optional)

### 2. CSV Batch Predictions

Create a CSV file with your bouts:

```csv
rikishi_a_id,rikishi_b_id,basho_id,day,rikishi_a_rank,rikishi_b_rank
123,456,630,1,5,8
789,321,630,5,2,10
```

Then run:

```bash
python3 predict_bouts.py --csv your_bouts.csv
```

Results will be saved to `your_bouts_predictions.csv`

### 3a. Single Prediction by Name (RECOMMENDED)

```bash
python3 predict_by_name.py --name1 "Abdelrahman Sharan" --name2 "Adiya Baasandorj" --basho 600 --day 10
```

You can use partial names - if multiple matches are found, you'll be prompted to select:

```bash
python3 predict_by_name.py --name1 "Akira" --name2 "Altan" --basho 600 --day 10
# Will show you all wrestlers with "Akira" in their name
```

### 3b. Single Prediction by ID

```bash
python3 predict_bouts.py --rikishi1 123 --rikishi2 456 --basho 630 --day 10
```

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

See `sample_bouts.csv` for an example.

## Understanding the Output

The prediction includes:

1. **Win Probabilities**: Percentage chance each wrestler wins
2. **Predicted Winner**: The wrestler more likely to win
3. **Confidence**: How confident the model is (50% = coin flip, 100% = certain)
4. **Fantasy League Points**: Expected value for fantasy league scoring
   - Shows expected points (probability × potential points)
   - Shows maximum points possible if wrestler wins
   - Based on upset potential (lower-ranked wrestler defeating higher-ranked)
5. **Key Features**: Important factors in the prediction:
   - `elo_diff`: Difference in Elo ratings (skill level)
   - `rank_diff`: Difference in rankings
   - `momentum_diff`: Recent form difference
   - `experience_diff`: Experience gap

### Fantasy League Scoring System

The fantasy points are calculated as follows:
- **Base points**: 2 points for any win
- **Upset points**: Additional points based on rank group difference when lower-ranked wrestler wins

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
- M13 defeats M5: 2 base + 2 upset = 4 points (2 groups lower)
- Sekiwake defeats Komusubi: 2 base + 0 upset = 2 points (higher rank wins)
- M12 defeats Yokozuna: 2 base + 6 upset = 8 points (major upset!)

**Expected Points**: The prediction shows expected value = win probability × potential points

## Model Details

- **Accuracy**: ~60.4% on historical data
- **Models**: Ensemble of Random Forest, LightGBM, and XGBoost
- **Training Data**: 40,718 historical bouts from bashos 491-630
- **Features**: 41 features including Elo ratings, rank, experience, momentum, win rates, head-to-head record, and more

## Re-training the Model

To re-train with updated data from the database:

```bash
python3 save_best_model.py
```

This will create a new `sumo_predictor_production.joblib` file with the latest data.

## Notes

- The model requires wrestlers to have historical bout data
- Predictions work best for wrestlers who have fought recently
- Rank information (if available) improves prediction accuracy
- The ~60% accuracy reflects the inherent unpredictability of sports
