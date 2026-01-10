# Sumo Handicapper - AI-Powered Bout Predictions

Machine learning system for predicting Japanese sumo wrestling bout outcomes with **60.4% accuracy** using ensemble models.

## Quick Start

All commands use `uv run` which works cross-platform on both Windows and macOS.

### Making Predictions

```bash
# Interactive prediction by wrestler name (easiest!)
uv run python predict_by_name.py --interactive

# Web UI (beautiful, interactive)
uv run streamlit run src/prediction/streamlit_app.py

# Single prediction
uv run python predict_by_name.py --name1 "Hakuho" --name2 "Asashoryu" --basho 630 --day 10

# Batch predictions from CSV
uv run python predict_bouts.py --csv data/sample_bouts.csv
```

### Training Models

```bash
# Train and save production model
uv run python -m src.training.save_best_model
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

This project uses modern Python packaging with `pyproject.toml` and `uv` for dependency management.

```bash
# Install uv if you haven't already
pip install uv

# Sync all dependencies
uv sync
```

## Documentation

- **[Setup Guide](docs/SETUP.md)** - Installation and database configuration
- **[Usage Guide](docs/USAGE.md)** - How to make predictions (CLI)
- **[Streamlit UI Guide](docs/STREAMLIT.md)** - Web interface documentation

## Model Details

- **Training**: 40,718 bouts (bashos 491-630)
- **Features**: 41 features (Elo, rank, momentum, h2h, etc.)
- **Ensemble**: RF (45%) + LGB (45%) + XGB (10%)

---

**Version**: 1.0 | **Accuracy**: 60.4% | **Updated**: November 2025
