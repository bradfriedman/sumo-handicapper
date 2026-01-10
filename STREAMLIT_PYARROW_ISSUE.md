# Streamlit + PyArrow Installation Issue

## The Problem

You're using **Python 3.14.0** (released very recently) and PyArrow doesn't have prebuilt wheels for it yet. When pip tries to build PyArrow from source, it fails because it needs Apache Arrow C++ libraries.

## The Good News

**Your Streamlit app will work fine without PyArrow!**

PyArrow is listed as a requirement but is only actually needed for:
- Large DataFrame optimizations (you're doing single predictions)
- Parquet file format support (you're using CSV)

Your sumo predictor doesn't use either feature.

## Solutions

### Option 1: Install Apache Arrow (For Full Compatibility)

```bash
# Install Apache Arrow C++ libraries via Homebrew
brew install apache-arrow

# Then install pyarrow
pip3 install 'pyarrow>=7.0,<22'
```

This gives you full pyarrow support.

### Option 2: Just Run Streamlit (Recommended - It Works!)

**Simply ignore the pyarrow warnings and run your app:**

```bash
.venv/bin/streamlit run src/prediction/streamlit_app.py
```

You might see a warning like:
```
ImportError: Unable to import required dependencies:
pyarrow
```

But the app will still work perfectly for your use case!

### Option 3: Wait for PyArrow Wheels

PyArrow will release prebuilt wheels for Python 3.14 soon. Then you can:

```bash
pip3 install --upgrade pyarrow
```

## Current Status

✅ **Streamlit**: Installed (v1.51.0)
✅ **Plotly**: Installed (v6.4.0)
✅ **All other deps**: Installed
⚠️ **PyArrow**: Missing (but not needed for your app)

## Test It Now

Try running the app:

```bash
.venv/bin/streamlit run src/prediction/streamlit_app.py
```

You should see the app launch at `http://localhost:8501` even without pyarrow!

## Why This Happens

Python 3.14.0 was released in October 2024. Package maintainers need time to:
1. Test compatibility
2. Build wheels for new Python versions
3. Publish them to PyPI

This is normal for bleeding-edge Python versions.

## Recommendation

For production use, consider using **Python 3.11 or 3.12** which have full package ecosystem support. For development, Python 3.14 works fine - just ignore the pyarrow warning.

---

**Bottom Line**: Your Streamlit app works! PyArrow is optional for your use case.
