# Sumo Handicapper - AI-Powered Bout Predictions

Machine learning system for predicting Japanese sumo wrestling bout outcomes with **60.4% accuracy** using ensemble models.

## Quick Start

### Making Predictions

```bash
# Interactive prediction by wrestler name (easiest!)
python3 predict_by_name.py --interactive

# Interactive prediction by ID
python3 predict_bouts.py --interactive

# Web UI (beautiful, interactive)
.venv/bin/streamlit run src/prediction/streamlit_app.py

# Single prediction
python3 predict_by_name.py --name1 "Hakuho" --name2 "Asashoryu" --basho 630 --day 10

# Batch predictions from CSV
python3 predict_bouts.py --csv data/sample_bouts.csv
```

### Training Models

```bash
# Train and save production model
python3 -m src.training.save_best_model
```

## Project Structure

```
sumo-handicapper/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── predict_by_name.py                # Wrapper: Name-based predictions
├── predict_bouts.py                  # Wrapper: ID-based predictions
├── run_streamlit.py                  # Wrapper: Launch Streamlit UI
│
├── models/                           # Trained models
│   ├── sumo_predictor_production.joblib
│   └── archive/                      # Old/experimental models
│
├── src/                              # Source code
│   ├── core/                         # Core ML components
│   │   ├── sumo_predictor.py         # Base predictor, Elo, data loader
│   │   └── fantasy_points.py         # Fantasy league scoring
│   │
│   ├── prediction/                   # Prediction interfaces
│   │   ├── prediction_engine.py      # Shared prediction logic
│   │   ├── predict_bouts.py          # CLI: ID-based predictions
│   │   ├── predict_by_name.py        # CLI: Name-based predictions
│   │   └── streamlit_app.py          # Web UI
│   │
│   ├── training/                     # Model training
│   │   ├── save_best_model.py        # Train production model
│   │   ├── enhanced_features.py      # Feature engineering
│   │   └── ...                       # Other training scripts
│   │
│   └── utils/                        # Utilities and tests
│
├── docs/                             # Documentation
│   ├── PREDICTION_README.md
│   ├── STREAMLIT_README.md
│   └── STREAMLIT_QUICKSTART.md
│
├── data/                             # Sample data
│   └── sample_bouts.csv
│
└── output/                           # Logs and results
    ├── logs/
    └── archive/
```

## Features

- **60.4% Accuracy** - Ensemble: Random Forest + LightGBM + XGBoost
- **Live Data** - Real-time h2h and recent form from database
- **Name Search** - Find wrestlers by partial/ring names
- **Web UI** - Interactive Streamlit interface with charts
- **Fantasy Points** - Calculate expected scores
- **Batch Processing** - CSV upload/download

## Installation

```bash
pip install -r requirements.txt

# For Streamlit UI (optional)
pip install streamlit plotly
```

## Documentation

- **[Prediction Guide](docs/PREDICTION_README.md)** - How to make predictions
- **[Streamlit UI Guide](docs/STREAMLIT_README.md)** - Web interface docs
- **[Quick Start](docs/STREAMLIT_QUICKSTART.md)** - Get started fast

## Model Details

- **Training**: 40,718 bouts (bashos 491-630)
- **Features**: 41 features (Elo, rank, momentum, h2h, etc.)
- **Ensemble**: RF (45%) + LGB (45%) + XGB (10%)

---

**Version**: 1.0 | **Accuracy**: 60.4% | **Updated**: November 2025
