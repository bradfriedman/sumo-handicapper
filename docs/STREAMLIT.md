# Streamlit Web UI Guide

Interactive web interface for predicting sumo wrestling bout outcomes.

## Features

- **Single Bout Predictions:** Search wrestlers by name or ID
- **Batch Predictions:** Upload CSV files for bulk predictions
- **Live Data:** Real-time head-to-head and recent form from database
- **Interactive Charts:** Win probabilities and model comparisons
- **Fantasy Points:** Expected fantasy league points calculator
- **Detailed Statistics:** Elo ratings, momentum, experience, and more

## Running the App

From the project directory:

```bash
uv run streamlit run src/prediction/streamlit_app.py
```

Opens automatically at http://localhost:8501

## Usage

### Single Bout Prediction Mode

1. **Select Mode:** "Single Bout Prediction" from sidebar

2. **Enter Bout Details:**
   - Basho ID (tournament, e.g., 630)
   - Day (1-15)

3. **Select Wrestlers:**
   - Choose **By Name** (recommended) or **By ID**
   - If by name: Enter full or partial name (e.g., "Hakuho")
   - If multiple matches: Select from dropdown
   - Repeat for second wrestler

4. **Get Prediction:** Click "Predict Bout Outcome"

### Results Display

**Winner Prediction:**
- Clear winner announcement
- Confidence level percentage

**Win Probabilities:**
- Visual bar chart
- Percentages for both wrestlers

**Head-to-Head Record:**
- Career matchup history (e.g., "21-36")
- Shows "0-0" for first-time matchups

**Detailed Statistics:**
- Key features (Elo difference, rank difference, momentum, experience)
- Individual model predictions (RF, LightGBM, XGBoost)

**Fantasy League Points:**
- Expected points (probability × potential)
- Maximum potential points

### Batch Predictions (CSV Upload)

1. **Select Mode:** "Batch Predictions (CSV)" from sidebar

2. **Prepare CSV:**
   ```csv
   rikishi_a_id,rikishi_b_id,basho_id,day
   30,245,630,10
   1,2,630,5
   ```

   Optional: `rikishi_a_rank`, `rikishi_b_rank`, `rikishi_a_dob`, `rikishi_b_dob`

3. **Upload:** Use file uploader

4. **Run:** Click "Run Batch Predictions"

5. **Download:** Get results as CSV

## Tips

1. **Name Search:** Partial names work (e.g., "Haku" finds "Hakuho")
2. **Ring Names:** Searches both real names and shikona (ring names)
3. **Basho Context:** Providing basho ID shows wrestler ranks for that tournament
4. **Model Confidence:** >70% = strong prediction, <55% = toss-up
5. **First Meetings:** Shows "0-0" for wrestlers who haven't fought

## Model Information

- **Accuracy:** ~60.4% on test data
- **Ensemble:** Random Forest (45%) + LightGBM (45%) + XGBoost (10%)
- **Training Data:** 40,718 bouts (bashos 491-630)
- **Features:** 41 features (Elo, rank, momentum, h2h, experience, etc.)

## Fantasy League Scoring

**Points Calculation:**
- Base: 2 points for any win
- Upset bonus: Points based on rank group difference

**Rank Groups:**
- Yokozuna (rank -3), Ozeki (-2), Sekiwake (-1), Komusubi (0)
- M1-4, M5-8, M9-M12, M13+

**Example:** M13 defeats M5 → 2 base + 2 upset = 4 points

## Architecture

The UI uses shared prediction logic:

```
src/prediction/prediction_engine.py → Shared logic (CLI + Web)
src/prediction/streamlit_app.py     → Web interface
src/core/fantasy_points.py          → Scoring calculations
src/core/sumo_predictor.py          → ML models
```

## Performance

- **Model Loading:** Cached after first load
- **Database Queries:** Real-time with optimized indexes
- **Batch Processing:** Progress bar for large CSVs

## Troubleshooting

**App won't start:**
- Verify Cloud SQL Proxy is running
- Check model file exists: `models/sumo_predictor_production.joblib`
- Ensure dependencies installed: `uv sync`

**Slow predictions:**
- Check database connection (proxy running?)
- First prediction loads model (subsequent ones are cached)

**CSV upload issues:**
- Verify CSV has required columns
- Check for proper formatting (commas, no extra spaces)

## Comparison with CLI

**Streamlit Advantages:**
- Beautiful visual interface
- Interactive charts
- No command-line knowledge required
- Real-time validation
- Easy CSV upload/download

**When to Use CLI:**
- Automation and scripting
- Integration with other tools
- CI/CD pipelines
- Batch processing without UI
