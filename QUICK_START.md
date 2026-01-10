# 🚀 Quick Start - Sumo Handicapper

## ✅ Everything is Ready!

Your project is fully set up with:
- ✅ Model trained (40,886 bouts, 60.4% accuracy)
- ✅ All code reorganized and working
- ✅ Streamlit installed and ready
- ✅ Modern Python packaging with pyproject.toml and uv

## 🔌 Database Setup (Cloud SQL Proxy)

This project connects to a Cloud SQL MySQL database. You need to run the Cloud SQL Proxy before using any features.

### Prerequisites

1. Install the Cloud SQL Auth Proxy:

**macOS:**
```bash
# Using Homebrew (recommended - always gets latest)
brew install cloud-sql-proxy

# Or download directly (v2.20.0)
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.20.0/cloud-sql-proxy.darwin.amd64
chmod +x cloud-sql-proxy
```

**Windows 11 (PowerShell):**
```powershell
# Download to user local bin directory (recommended - follows XDG standard)
New-Item -ItemType Directory -Path "$env:USERPROFILE\.local\bin" -Force
Invoke-WebRequest -Uri "https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.20.0/cloud-sql-proxy.x64.exe" -OutFile "$env:USERPROFILE\.local\bin\cloud-sql-proxy.exe"

# Add to PATH (restart PowerShell after running this)
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$env:USERPROFILE\.local\bin*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$env:USERPROFILE\.local\bin", "User")
}

# Or download to current directory only
# Invoke-WebRequest -Uri "https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.20.0/cloud-sql-proxy.x64.exe" -OutFile "cloud-sql-proxy.exe"
```

> **Note:** For the latest version, check the [Cloud SQL Proxy releases page](https://github.com/GoogleCloudPlatform/cloud-sql-proxy/releases)

2. Authenticate with Google Cloud:

**macOS/Linux:**
```bash
gcloud auth application-default login
```

**Windows 11 (PowerShell):**
```powershell
gcloud auth application-default login
``` 

### Starting the Proxy

Run this command in a separate terminal window and keep it running:

**macOS/Linux:**
```bash
cloud-sql-proxy <YOUR_INSTANCE_CONNECTION_NAME> --port 3307
```

**Windows 11 (PowerShell):**
```powershell
# If installed to PATH (recommended setup above)
cloud-sql-proxy <YOUR_INSTANCE_CONNECTION_NAME> --port 3307

# Or if downloaded to current directory only
# .\cloud-sql-proxy.exe <YOUR_INSTANCE_CONNECTION_NAME> --port 3307
```

Replace `<YOUR_INSTANCE_CONNECTION_NAME>` with your Cloud SQL instance connection name (format: `project:region:instance`).

**Example (macOS/Linux):**
```bash
cloud-sql-proxy my-project:us-central1:dewsweeper-db --port 3307
```

**Example (Windows 11):**
```powershell
cloud-sql-proxy my-project:us-central1:dewsweeper-db --port 3307
```

### Verifying the Connection

The proxy should display:
```
Listening on 127.0.0.1:3307
Ready for new connections
```

Your application is configured to connect to:
- Host: `127.0.0.1`
- Port: `3307`
- Database: `dewsweeper3`
- User: `dewsweeper`

See [src/prediction/prediction_engine.py:13-19](src/prediction/prediction_engine.py#L13-L19) for the connection configuration.

### Troubleshooting Proxy Issues

**"Cannot connect to Cloud SQL instance"**
- Ensure you've run `gcloud auth application-default login`
- Verify your GCP account has Cloud SQL Client role
- Check that the instance connection name is correct

**"Port 3307 already in use"**
- macOS/Linux: Stop any existing proxy instances: `pkill cloud-sql-proxy`
- Windows: Stop via Task Manager or: `Stop-Process -Name cloud-sql-proxy`
- Or use a different port and update DB_CONFIG in the code

**"Connection refused"**
- Make sure the proxy is running before starting the application
- Check that the proxy shows "Ready for new connections"

---

## 📝 Quick Commands

### Make Predictions

**macOS/Linux:**
```bash
# Interactive prediction by name (EASIEST!)
python3 predict_by_name.py --interactive

# Interactive prediction by ID
python3 predict_bouts.py --interactive

# Web UI (beautiful interface)
.venv/bin/streamlit run src/prediction/streamlit_app.py

# Single prediction
python3 predict_bouts.py --rikishi1 30 --rikishi2 245 --basho 630 --day 10

# Batch from CSV
python3 predict_bouts.py --csv data/sample_bouts.csv
```

**Windows 11 (PowerShell):**
```powershell
# Interactive prediction by name (EASIEST!)
.venv\Scripts\python.exe predict_by_name.py --interactive

# Interactive prediction by ID
.venv\Scripts\python.exe predict_bouts.py --interactive

# Web UI (beautiful interface)
.venv\Scripts\streamlit.exe run src/prediction/streamlit_app.py

# Single prediction
.venv\Scripts\python.exe predict_bouts.py --rikishi1 30 --rikishi2 245 --basho 630 --day 10

# Batch from CSV
.venv\Scripts\python.exe predict_bouts.py --csv data/sample_bouts.csv
```

### Alternative Streamlit Commands

**macOS/Linux:**
```bash
# Option 1: Via Python module
python3 -m streamlit run src/prediction/streamlit_app.py

# Option 2: Activate venv first
source .venv/bin/activate
streamlit run src/prediction/streamlit_app.py
```

**Windows 11 (PowerShell):**
```powershell
# Option 1: Via Python module
.venv\Scripts\python.exe -m streamlit run src/prediction/streamlit_app.py

# Option 2: Activate venv first
.venv\Scripts\Activate.ps1
streamlit run src/prediction/streamlit_app.py
```

### Retrain Model (if needed)

**macOS/Linux:**
```bash
python3 -m src.training.save_best_model
```

**Windows 11 (PowerShell):**
```powershell
.venv\Scripts\python.exe -m src.training.save_best_model
```

## 📊 What You'll See

### CLI Predictions
```
Loading prediction model...
Rikishi 30 win probability: 25.4%
Rikishi 245 win probability: 74.6%

🎯 Predicted winner: Rikishi 245
   Confidence: 74.6%

📊 Career Head-to-Head: 21-36
```

### Streamlit UI
- Opens at `http://localhost:8501`
- Search wrestlers by name
- Interactive charts
- Fantasy points calculator
- Batch CSV upload/download

## 📁 Project Structure

```
sumo-handicapper/
├── models/          # Trained models
├── src/
│   ├── core/       # ML core
│   ├── prediction/ # Interfaces
│   └── training/   # Training
├── docs/           # Documentation
└── data/           # Sample data
```

## 💡 Tips

1. **Always run from project root**:
   - macOS: `/Users/brad/Projects/sumo-handicapper/`
   - Windows: `C:\Users\brad.friedman\Projects\sumo-handicapper\`
2. **PyArrow warning?** Ignore it - your app works fine without it
3. **Names work best**: Use partial names like "Haku" to find "Hakuho"
4. **CSV format**: See `data/sample_bouts.csv` for example
5. **Dependencies**: This project now uses `pyproject.toml` and `uv.lock` for modern Python dependency management

## 🆘 Troubleshooting

### "Cannot connect to database" or connection errors
Make sure the Cloud SQL Proxy is running (see Database Setup section above)

### "streamlit: command not found" (or not recognized)
- macOS: Use `.venv/bin/streamlit run src/prediction/streamlit_app.py`
- Windows: Use `.venv\Scripts\streamlit.exe run src/prediction/streamlit_app.py`

### "ModuleNotFoundError"
Make sure you're in the project root directory

### "Model file not found"
- macOS: Run `python3 -m src.training.save_best_model`
- Windows: Run `.venv\Scripts\python.exe -m src.training.save_best_model`

## 📚 Full Documentation

- `README.md` - Complete project overview
- `docs/PREDICTION_README.md` - Prediction guide
- `docs/STREAMLIT_README.md` - Web UI guide
- `REORGANIZATION_SUMMARY.md` - What changed
- `STREAMLIT_PYARROW_ISSUE.md` - PyArrow explained

---

**Ready to predict!**

Start with:
- macOS: `python3 predict_by_name.py --interactive`
- Windows: `.venv\Scripts\python.exe predict_by_name.py --interactive`
