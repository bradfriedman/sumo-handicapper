# Streamlit UI - Quick Start Guide

## You're Ready to Run!

All required packages are already installed. The Streamlit UI is ready to use.

## Required Packages ✅

All installed and verified:
- ✅ streamlit (1.51.0)
- ✅ plotly (6.4.0)
- ✅ pandas (2.3.3)
- ✅ numpy (2.3.4)
- ✅ pymysql (1.1.2)
- ✅ joblib (1.5.2)
- ✅ scikit-learn (1.7.2)
- ✅ lightgbm (4.6.0)
- ✅ xgboost (3.1.1)

## Run the App

From your project directory:

**macOS/Linux:**
```bash
streamlit run streamlit_app.py
# Or with full path
.venv/bin/streamlit run src/prediction/streamlit_app.py
```

**Windows 11 (PowerShell):**
```powershell
streamlit run streamlit_app.py
# Or with full path
.venv\Scripts\streamlit.exe run src\prediction\streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## What You Can Do

### 1. Single Bout Predictions
- Search wrestlers by **name** (e.g., "Hakuho") or **ID**
- Get win probabilities with confidence levels
- See head-to-head career records
- View fantasy league points

### 2. Batch Predictions
- Upload CSV files with multiple bouts
- Download results as CSV

### 3. Interactive Visualizations
- Win probability bar charts
- Individual model comparison charts
- Real-time data from database

## Features

- 🎯 **60.4% Accuracy** - Ensemble ML model
- 📡 **Live Data** - Real-time h2h and recent records
- 📊 **Visual Charts** - Interactive Plotly graphs
- 🎮 **Fantasy Points** - Expected value calculations
- 🔍 **Name Search** - Partial matching, ring names
- 📥 **CSV Upload** - Batch predictions

## Tips

1. **Name Search**: Type partial names like "Haku" to find "Hakuho"
2. **Ring Names**: Searches both real names and shikona (ring names)
3. **Basho Context**: Providing basho ID shows wrestler ranks
4. **Model Confidence**: >70% confidence = strong prediction
5. **First Meetings**: Shows "0-0" for wrestlers who haven't fought

## Note About PyArrow Warning

You may see a warning about `pyarrow` when starting the app. This is **completely safe to ignore**.
The app works perfectly without it.

If you want to install it (optional):

**macOS:**
```bash
# Install Apache Arrow C++ libraries first
brew install apache-arrow

# Then install pyarrow
pip install 'pyarrow>=7.0,<22'
```

**Windows 11:**
```powershell
# Install pyarrow directly (pre-built wheels available)
pip install 'pyarrow>=7.0,<22'
```

## Architecture

The app uses a clean, refactored architecture:

```
prediction_engine.py  → Shared prediction logic
streamlit_app.py      → Web UI
fantasy_points.py     → Scoring calculations
sumo_predictor.py     → ML models and features
```

This means the same prediction code powers both the CLI scripts and web UI!

## Files

- `streamlit_app.py` - Main Streamlit application
- `prediction_engine.py` - Shared prediction logic
- `STREAMLIT_README.md` - Detailed documentation
- `STREAMLIT_QUICKSTART.md` - This file
- `streamlit_requirements.txt` - Package list

## Support

If you encounter issues:
1. Verify model exists:
   - macOS/Linux: `ls sumo_predictor_production.joblib`
   - Windows: `dir sumo_predictor_production.joblib` or `ls sumo_predictor_production.joblib` (in PowerShell)
2. Check database connection (should auto-connect)
3. Restart Streamlit if needed: Ctrl+C, then rerun

## Enjoy!

You now have a beautiful web interface for your sumo predictions. Happy predicting! 🥋
