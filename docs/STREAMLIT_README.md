# Sumo Bout Predictor - Streamlit UI

A beautiful, reactive web interface for predicting sumo wrestling bout outcomes using machine learning.

## Features

- **Single Bout Predictions**: Predict individual matchups by wrestler name or ID
- **Batch Predictions**: Upload CSV files for bulk predictions
- **Live Data**: Real-time head-to-head and recent form data from the database
- **Interactive Visualizations**: Charts showing win probabilities and model comparisons
- **Fantasy Points**: Expected fantasy league points based on upset potential
- **Detailed Statistics**: Elo ratings, momentum, experience, and more

## Installation

### Required Packages

Install the required packages using pip:

```bash
pip3 install streamlit plotly
```

### Additional Dependencies

The following packages should already be installed if you've set up the prediction system:

- pandas
- numpy
- pymysql
- joblib
- scikit-learn
- lightgbm
- xgboost

### Troubleshooting PyArrow (Optional)

Streamlit has a dependency on `pyarrow` which may fail to build on some systems (especially Python 3.14). You have two options:

**Option 1:** Install cmake (required to build pyarrow)
```bash
# On macOS with Homebrew
brew install cmake

# Then install pyarrow
pip3 install 'pyarrow<22,>=7.0'
```

**Option 2:** Run without pyarrow (basic functionality will still work)
The app will run fine without pyarrow for most use cases. You may see a warning but can ignore it.

## Running the App

From the project directory, run:

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage Guide

### Single Bout Prediction Mode

1. **Select Prediction Mode**: Choose "Single Bout Prediction" from the sidebar

2. **Enter Bout Details**:
   - Basho ID (tournament identifier, e.g., 630)
   - Day (1-15)

3. **Select Rikishi**:
   - Choose search method: **By Name** (recommended) or **By ID**
   - If by name: Enter full or partial name (e.g., "Hakuho")
   - If multiple matches found, select from dropdown
   - Repeat for second wrestler

4. **Get Prediction**: Click "Predict Bout Outcome" button

### Results Displayed

The prediction results include:

- **Winner Prediction**: Clear winner announcement with confidence level
- **Win Probabilities**: Visual chart and percentages for both wrestlers
- **Head-to-Head Record**: Career matchup history
- **Detailed Statistics**:
  - Key features (Elo difference, rank difference, experience, momentum)
  - Individual model predictions (Random Forest, LightGBM, XGBoost)
- **Fantasy League Points**: Expected points and maximum potential points

### Batch Predictions (CSV Upload)

1. **Select Mode**: Choose "Batch Predictions (CSV)" from sidebar

2. **Prepare CSV File** with required columns:
   ```csv
   rikishi_a_id,rikishi_b_id,basho_id,day
   30,245,630,10
   1,2,630,5
   ```

   Optional columns: `rikishi_a_rank`, `rikishi_b_rank`, `rikishi_a_dob`, `rikishi_b_dob`

3. **Upload CSV**: Use the file uploader to select your CSV

4. **Run Predictions**: Click "Run Batch Predictions"

5. **Download Results**: Use the download button to get results as CSV

## Architecture

The Streamlit UI uses a refactored architecture:

### `prediction_engine.py`
Shared prediction logic used by both CLI scripts and Streamlit UI:
- `load_model()`: Load trained model with live data enabled
- `predict_bout()`: Make predictions for a single bout
- `search_rikishi_by_name()`: Search for wrestlers by name
- `get_rikishi_by_id()`: Look up wrestler by ID

### `streamlit_app.py`
Web interface built with:
- **Streamlit**: Reactive web framework
- **Plotly**: Interactive charts and visualizations
- **Custom CSS**: Beautiful gradient styling

### Live Data Queries

The app queries fresh data from the database at prediction time:
- Head-to-head career records
- Recent bout results (momentum)
- Current basho records

## Model Information

- **Ensemble Model**: Combines Random Forest, LightGBM, and XGBoost
- **Training Data**: 40,718 historical bouts (bashos 491-630)
- **Accuracy**: ~60.4% on test data
- **Features**: 41 features including Elo ratings, rank, experience, momentum, win rates, and head-to-head records

## Fantasy League Scoring

The app calculates fantasy points based on:

- **Base Points**: 2 points for any win
- **Upset Bonus**: Additional points when lower-ranked wrestler wins
- **Expected Points**: Win probability × Potential points

**Rank Groups**:
- Yokozuna (rank -3)
- Ozeki (rank -2)
- Sekiwake (rank -1)
- Komusubi (rank 0)
- M1-4 (ranks 1-4)
- M5-8 (ranks 5-8)
- M9-M12 (ranks 9-12)
- M13+ (ranks 13-20)

**Example**: M13 defeats M5 → 2 base + 2 upset = 4 points

## Tips

1. **Name Search**: You can use partial names (e.g., "Haku" will find "Hakuho")
2. **Ring Names**: The system searches both real names and ring names (shikona)
3. **Basho Context**: When you provide a basho ID, the system shows wrestler ranks for that tournament
4. **Model Confidence**: Higher confidence (>70%) indicates stronger predictions
5. **First Meetings**: Head-to-head shows "0-0" for wrestlers who haven't fought before

## Performance

- **Model Loading**: Cached after first load (fast subsequent predictions)
- **Database Queries**: Real-time queries optimized with indexes
- **Batch Processing**: Progress bar shows status for large CSV files

## Support

For issues or questions:
- Check that the model file `sumo_predictor_production.joblib` exists
- Verify database connection (host: 127.0.0.1, port: 3307)
- Ensure all required packages are installed

## Files Created

- `prediction_engine.py`: Shared prediction logic
- `streamlit_app.py`: Streamlit web interface
- `STREAMLIT_README.md`: This file

## Comparison with CLI Scripts

The Streamlit UI provides the same functionality as:
- `predict_by_name.py`: Interactive name-based predictions
- `predict_bouts.py`: ID-based and CSV batch predictions

**Advantages of Streamlit UI**:
- Beautiful visual interface
- Interactive charts and graphs
- No command-line knowledge required
- Real-time validation
- Easy CSV upload/download

**When to Use CLI**:
- Automation and scripting
- Integration with other tools
- Running predictions in CI/CD pipelines
