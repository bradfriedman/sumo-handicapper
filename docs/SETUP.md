# Setup Guide - Sumo Handicapper

Complete setup instructions for running the sumo bout prediction system on macOS or Windows.

## Prerequisites

- Python 3.12+
- Google Cloud SQL Proxy (for database access)
- `uv` for dependency management (recommended) or `pip`

## 1. Installation

### Install Dependencies

**Using uv (recommended):**
```bash
# Install uv if needed
pip install uv

# Sync all dependencies
uv sync
```

**Using pip:**
```bash
pip install -r requirements.txt
```

## 2. Database Setup (Cloud SQL Proxy)

This project connects to a Cloud SQL MySQL database via proxy.

### Install Cloud SQL Proxy

**macOS:**
```bash
# Using Homebrew (recommended)
brew install cloud-sql-proxy
```

**Windows:**
```powershell
# Download to user local bin
New-Item -ItemType Directory -Path "$env:USERPROFILE\.local\bin" -Force
Invoke-WebRequest -Uri "https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.20.0/cloud-sql-proxy.x64.exe" -OutFile "$env:USERPROFILE\.local\bin\cloud-sql-proxy.exe"

# Add to PATH (restart PowerShell after)
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$env:USERPROFILE\.local\bin*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$env:USERPROFILE\.local\bin", "User")
}
```

### Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

### Start the Proxy

Run in a separate terminal (keep it running):

```bash
cloud-sql-proxy <YOUR_INSTANCE_CONNECTION_NAME> --port 3307
```

Replace `<YOUR_INSTANCE_CONNECTION_NAME>` with your instance (format: `project:region:instance`).

**Example:**
```bash
cloud-sql-proxy my-project:us-central1:dewsweeper-db --port 3307
```

Verify you see:
```
Listening on 127.0.0.1:3307
Ready for new connections
```

### Connection Details

- **Host:** 127.0.0.1
- **Port:** 3307
- **Database:** dewsweeper3
- **User:** dewsweeper

## 3. Quick Start

### Run Predictions

**Interactive mode (recommended):**
```bash
uv run python predict_by_name.py --interactive
```

**Web UI:**
```bash
uv run streamlit run src/prediction/streamlit_app.py
```

Opens at http://localhost:8501

**Single prediction:**
```bash
uv run python predict_by_name.py --name1 "Hakuho" --name2 "Asashoryu" --basho 630 --day 10
```

**Batch predictions from CSV:**
```bash
uv run python predict_bouts.py --csv data/sample_bouts.csv
```

### Retrain Model

If you need to retrain with fresh data:

```bash
uv run python -m src.training.save_best_model
```

## Troubleshooting

### Database Connection Issues

**"Cannot connect to database"**
- Ensure Cloud SQL Proxy is running
- Run `gcloud auth application-default login`
- Verify GCP account has Cloud SQL Client role

**"Port 3307 already in use"**
- Stop existing proxy: `pkill cloud-sql-proxy` (macOS) or Task Manager (Windows)
- Or use different port and update DB_CONFIG in code

### Module Not Found Errors

- Ensure you're in project root directory
- Run `uv sync` to reinstall dependencies

### Model File Not Found

- Run the training command above to generate the model

## Tips

1. **Always run from project root:** `C:\Users\brad.friedman\Projects\sumo-handicapper\`
2. **Use `uv run` prefix:** Works cross-platform without activating venv
3. **Names work best:** Use partial names like "Haku" to find "Hakuho"
4. **CSV format:** See `data/sample_bouts.csv` for example

## Project Structure

```
sumo-handicapper/
├── models/          # Trained models (.joblib files)
├── src/
│   ├── core/       # ML core components
│   ├── prediction/ # Prediction interfaces (CLI, web)
│   ├── training/   # Model training scripts
│   └── utils/      # Utilities and tests
├── docs/           # Documentation (you are here)
├── data/           # Sample data
└── output/         # Logs and results
```

## Next Steps

See:
- [docs/USAGE.md](USAGE.md) - Detailed usage examples
- [docs/STREAMLIT.md](STREAMLIT.md) - Web UI guide
- [README.md](../README.md) - Project overview
