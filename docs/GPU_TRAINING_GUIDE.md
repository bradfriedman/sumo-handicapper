# GPU Training Guide

This guide explains how to leverage your NVIDIA GPU for faster model training in the Sumo Handicapper project.

## Current Status

### ✅ What's Working
- **GPU Detection**: Automatically detects NVIDIA GPUs and their capabilities
- **XGBoost GPU Acceleration**: Fully functional with CUDA support
- **Automatic Configuration**: Training scripts now automatically use GPU when available
- **Adaptive Optimization**: Adjusts parameters based on GPU memory (entry-level, mid-range, or high-end)

### ⚠️ What Needs Configuration
- **LightGBM GPU Support**: Requires GPU-enabled build (see instructions below)

## Performance Improvements

High-end NVIDIA GPUs provide:
- **10-50x faster** training compared to CPU for large datasets (500K+ samples)
- Ability to train **larger models** (more trees, deeper trees) efficiently
- Can handle **500-1000 estimators** for better accuracy
- **Note**: For the current dataset (~82K samples), CPU is actually faster.

## Quick Start

### 1. Test GPU Detection

```bash
uv run python src/utils/gpu_optimizer.py
```

This will show you:
- GPU name and memory
- CUDA compute capability
- Which frameworks have GPU support
- Optimized parameter recommendations

### 2. Train Models with GPU

The training scripts now automatically use GPU acceleration:

```bash
# Update model with GPU acceleration
uv run python src/training/update_model.py

# Train from scratch
uv run python src/training/save_best_model.py
```

The scripts will:
- Detect your NVIDIA GPU and memory
- Enable GPU acceleration for XGBoost (if beneficial)
- Adjust n_estimators based on GPU capabilities
- Use optimized parameters for your hardware

## Enabling LightGBM GPU Support (Optional)

LightGBM requires a special GPU-enabled build. The current version uses CPU, which is still fast.

### Option 1: Build from Source (Recommended for Windows)

```bash
# Install dependencies
uv pip install cmake

# Clone and build LightGBM with GPU support
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake -A x64 -DUSE_GPU=1 ..
cmake --build . --target _lightgbm --config Release

# Install the built package
cd ../python-package
uv pip install --no-deps .
```

### Option 2: Use Pre-built GPU Version (Linux)

```bash
uv pip uninstall lightgbm
uv pip install lightgbm --install-option=--gpu
```

### Option 3: Stick with CPU LightGBM

LightGBM is already very fast on CPU, so this is optional. The ensemble model will still benefit from XGBoost GPU acceleration.

## Technical Details

### XGBoost GPU Optimizations

The GPU optimizer automatically applies settings based on your GPU:

**High-end GPU (>20GB VRAM):**
```python
{
    'device': 'cuda',
    'tree_method': 'hist',
    'max_bin': 512,
    'grow_policy': 'depthwise',
    'n_estimators': 500
}
```

**Mid-range GPU (10-20GB VRAM):**
```python
{
    'device': 'cuda',
    'tree_method': 'hist',
    'max_bin': 384,
    'grow_policy': 'depthwise',
    'n_estimators': 400
}
```

**Entry-level GPU (<10GB VRAM):**
```python
{
    'device': 'cuda',
    'tree_method': 'hist',
    'max_bin': 256,
    'n_estimators': 300
}
```

### Expected Speedup

| Model     | CPU Time (est.) | GPU Time (est.) | Speedup |
|-----------|-----------------|-----------------|---------|
| XGBoost   | 10-15 min       | 30-60 sec       | 10-20x  |
| LightGBM* | 5-8 min         | 20-40 sec       | 10-15x  |
| RF        | 3-5 min         | 3-5 min         | 1x      |

*LightGBM GPU times only if GPU-enabled build is installed

### Memory Usage

GPU memory requirements for this project:
- Typical training dataset: ~2-3GB
- Model memory: <500MB
- **Minimum recommended**: 6GB VRAM
- **Optimal**: 10GB+ VRAM for larger models

## Troubleshooting

### GPU Not Detected

```bash
# Check if nvidia-smi works
nvidia-smi
```

If this fails, update your NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx

### XGBoost Not Using GPU

The current XGBoost installation should support GPU. If it doesn't:

```bash
# Check XGBoost build
uv run python -c "import xgboost; print(xgboost.build_info())"

# Look for "USE_CUDA": "ON" in the output
```

### CUDA Out of Memory

Unlikely with 24GB VRAM, but if it happens:
- Reduce `n_estimators`
- Reduce `max_bin`
- Process data in smaller batches

## Monitoring GPU Usage

While training, monitor GPU usage in another terminal:

```bash
# Watch GPU utilization in real-time
nvidia-smi -l 1
```

You should see:
- GPU utilization: 80-100%
- Memory usage: 2-5GB
- Temperature: 60-80°C (normal under load)

## Code Integration

The GPU optimizer is now integrated into all training scripts. To use it in your own scripts:

```python
from src.utils.gpu_optimizer import GPUOptimizer

# Initialize optimizer
gpu_opt = GPUOptimizer()
gpu_opt.print_summary()

# Get optimized parameters
xgb_params = gpu_opt.get_xgboost_params({
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200  # Will be adjusted based on GPU
})

lgb_params = gpu_opt.get_lightgbm_params({
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200
})

# Create models with optimized params
import xgboost as xgb
model = xgb.XGBClassifier(**xgb_params)
```

## Next Steps

1. **Test the optimizer** to verify GPU detection
2. **Run training** with the default settings (GPU acceleration is automatic)
3. **Monitor performance** with nvidia-smi
4. **(Optional) Enable LightGBM GPU** for maximum performance

The GPU optimizer will automatically configure optimal settings for your specific GPU!
