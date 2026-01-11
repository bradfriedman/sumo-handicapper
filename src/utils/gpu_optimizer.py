"""
GPU Detection and Optimization for Model Training

Automatically detects available GPU hardware and configures optimal training parameters
for XGBoost and LightGBM on NVIDIA GPUs (especially RTX 3090).
"""
import platform
import subprocess
import warnings
from typing import Dict, Optional, Tuple


class GPUOptimizer:
    """Detects GPU capabilities and provides optimized training configurations"""

    def __init__(self):
        self.gpu_available = False
        self.gpu_name = None
        self.gpu_memory_mb = None
        self.cuda_available = False
        self.compute_capability = None

        self._detect_gpu()

    def _detect_gpu(self):
        """Detect NVIDIA GPU and CUDA availability"""
        # Check for CUDA using nvidia-smi
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output:
                        parts = output.split(',')
                        self.gpu_name = parts[0].strip()
                        self.gpu_memory_mb = int(float(parts[1].strip().split()[0]))
                        self.compute_capability = parts[2].strip()
                        self.gpu_available = True
                        print(f"[OK] GPU Detected: {self.gpu_name}")
                        print(f"  Memory: {self.gpu_memory_mb} MB")
                        print(f"  Compute Capability: {self.compute_capability}")
            else:
                # Linux/Mac
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output:
                        parts = output.split(',')
                        self.gpu_name = parts[0].strip()
                        self.gpu_memory_mb = int(float(parts[1].strip().split()[0]))
                        self.gpu_available = True
                        print(f"[OK] GPU Detected: {self.gpu_name}")
                        print(f"  Memory: {self.gpu_memory_mb} MB")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        # Check if XGBoost has GPU support
        if self.gpu_available:
            try:
                import xgboost as xgb
                # Try to create a GPU-enabled booster
                test_params = {'tree_method': 'hist', 'device': 'cuda'}
                dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bst = xgb.train(test_params, dtrain, num_boost_round=1)
                self.cuda_available = True
                print("[OK] XGBoost GPU support: Available")
            except Exception:
                print("[WARN] XGBoost GPU support: Not available (CPU build)")
                print("  To enable: pip install xgboost[gpu] or build from source")

    def get_xgboost_params(self, base_params: Optional[Dict] = None) -> Dict:
        """
        Get optimized XGBoost parameters for detected hardware

        Args:
            base_params: Base parameters to merge with GPU optimizations

        Returns:
            Optimized parameter dictionary
        """
        params = base_params.copy() if base_params else {}

        if self.cuda_available:
            # GPU-specific optimizations
            params['device'] = 'cuda'
            params['tree_method'] = 'hist'  # GPU-accelerated histogram algorithm

            # Optimize based on GPU memory
            # High-end GPUs (>20GB VRAM) can handle more bins and deeper trees
            if self.gpu_memory_mb and self.gpu_memory_mb > 20000:
                params['max_bin'] = 512  # More bins for better accuracy with high-end GPU
                params['grow_policy'] = 'depthwise'  # Better for GPU
                print("\n[OK] High-end GPU Optimizations Applied:")
                print("  - GPU acceleration enabled (device=cuda)")
                print("  - Histogram tree method (tree_method=hist)")
                print("  - Increased bins for better accuracy (max_bin=512)")
            elif self.gpu_memory_mb and self.gpu_memory_mb > 10000:
                params['max_bin'] = 384  # Mid-range GPUs
                params['grow_policy'] = 'depthwise'
                print("\n[OK] Mid-range GPU Optimizations Applied:")
                print("  - GPU acceleration enabled (device=cuda)")
                print("  - Histogram tree method (tree_method=hist)")
                print("  - Optimized bins (max_bin=384)")
            else:
                params['max_bin'] = 256  # Entry-level GPUs

            print(f"\n[OK] XGBoost GPU Parameters:")
            for key in ['device', 'tree_method', 'max_bin', 'grow_policy']:
                if key in params:
                    print(f"  {key}: {params[key]}")
        else:
            # CPU fallback
            params['tree_method'] = 'hist'  # Still faster on CPU
            params['n_jobs'] = -1  # Use all CPU cores
            print("\n[INFO] XGBoost CPU Parameters (GPU not available):")
            print(f"  tree_method: hist")
            print(f"  n_jobs: -1 (all cores)")

        return params

    def get_lightgbm_params(self, base_params: Optional[Dict] = None) -> Dict:
        """
        Get optimized LightGBM parameters for detected hardware

        Args:
            base_params: Base parameters to merge with GPU optimizations

        Returns:
            Optimized parameter dictionary
        """
        params = base_params.copy() if base_params else {}

        if self.gpu_available:
            try:
                import lightgbm as lgb
                # Check if LightGBM has GPU support
                try:
                    # Create test dataset
                    import numpy as np
                    X_test = np.random.rand(100, 10)
                    y_test = np.random.randint(0, 2, 100)
                    train_data = lgb.Dataset(X_test, label=y_test)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        test_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0, 'verbose': -1}
                        lgb.train(test_params, train_data, num_boost_round=1)

                    # GPU is available
                    params['device'] = 'gpu'
                    params['gpu_platform_id'] = 0
                    params['gpu_device_id'] = 0

                    # LightGBM GPU optimizations
                    # Note: LightGBM GPU has max_bin limit of 255
                    params['gpu_use_dp'] = False  # Use single precision for speed
                    params['max_bin'] = 255  # Max allowed for GPU
                    if self.gpu_memory_mb and self.gpu_memory_mb > 20000:
                        print("\n[OK] LightGBM GPU Parameters (High-end GPU):")
                    else:
                        print("\n[OK] LightGBM GPU Parameters:")

                    for key in ['device', 'gpu_platform_id', 'gpu_device_id', 'max_bin', 'gpu_use_dp']:
                        if key in params:
                            print(f"  {key}: {params[key]}")

                except Exception as e:
                    # GPU support not available
                    params['device'] = 'cpu'
                    params['n_jobs'] = -1
                    print(f"\n[WARN] LightGBM GPU support: Not available")
                    print(f"  Reason: {str(e)}")
                    print("  To enable: Reinstall with GPU support (pip install lightgbm --install-option=--gpu)")
                    print(f"  Using CPU with all cores (n_jobs=-1)")
            except ImportError:
                print("\n[WARN] LightGBM not installed")
        else:
            # CPU fallback
            params['device'] = 'cpu'
            params['n_jobs'] = -1
            print("\n[INFO] LightGBM CPU Parameters (GPU not detected):")
            print(f"  device: cpu")
            print(f"  n_jobs: -1 (all cores)")

        return params

    def get_training_recommendations(self) -> Dict[str, any]:
        """
        Get overall training recommendations based on hardware

        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'use_gpu': self.cuda_available,
            'gpu_name': self.gpu_name,
            'gpu_memory_mb': self.gpu_memory_mb,
            'recommended_batch_size': None,
            'recommended_n_estimators': None,
        }

        if self.cuda_available:
            # Recommendations based on GPU memory
            if self.gpu_memory_mb and self.gpu_memory_mb > 20000:
                # High-end GPU (20GB+ VRAM)
                recommendations['recommended_batch_size'] = 'auto'
                recommendations['recommended_n_estimators'] = 500
                recommendations['notes'] = [
                    f"High-end GPU detected ({self.gpu_memory_mb}MB VRAM) - excellent for gradient boosting!",
                    "GPU acceleration will provide significant speedup for large datasets",
                    "Can train larger models (more trees, deeper trees) efficiently",
                    "Consider increasing n_estimators to 500-1000 for better accuracy"
                ]
            elif self.gpu_memory_mb and self.gpu_memory_mb > 10000:
                # Mid-range GPU (10-20GB VRAM)
                recommendations['recommended_batch_size'] = 'auto'
                recommendations['recommended_n_estimators'] = 400
                recommendations['notes'] = [
                    f"Mid-range GPU detected ({self.gpu_memory_mb}MB VRAM)",
                    "GPU acceleration will help with larger datasets",
                    "Can train moderately complex models efficiently"
                ]
            else:
                # Entry-level GPU (<10GB VRAM)
                recommendations['recommended_batch_size'] = 'auto'
                recommendations['recommended_n_estimators'] = 300
                recommendations['notes'] = [
                    f"GPU detected ({self.gpu_memory_mb}MB VRAM)",
                    "GPU acceleration available for moderate workloads",
                    "May need to adjust batch sizes for memory constraints"
                ]
        else:
            recommendations['recommended_batch_size'] = 'auto'
            recommendations['recommended_n_estimators'] = 200
            recommendations['notes'] = [
                "No GPU detected - using CPU",
                "Training will be slower, but still functional",
                "Consider keeping n_estimators around 200-300 for reasonable training time"
            ]

        return recommendations

    def print_summary(self):
        """Print a summary of GPU detection and optimization"""
        print("\n" + "="*80)
        print("GPU OPTIMIZATION SUMMARY")
        print("="*80)

        if self.gpu_available:
            print(f"\n[OK] GPU: {self.gpu_name}")
            print(f"  Memory: {self.gpu_memory_mb} MB")
            if self.compute_capability:
                print(f"  Compute Capability: {self.compute_capability}")

            print(f"\n[OK] CUDA for XGBoost: {'Available' if self.cuda_available else 'Not Available'}")

            recommendations = self.get_training_recommendations()
            if recommendations['notes']:
                print("\nRecommendations:")
                for note in recommendations['notes']:
                    print(f"  - {note}")
        else:
            print("\n[INFO] No NVIDIA GPU detected")
            print("  Training will use CPU (slower but functional)")

        print("="*80 + "\n")


def get_optimized_training_params() -> Tuple[Dict, Dict]:
    """
    Convenience function to get optimized parameters for both XGBoost and LightGBM

    Returns:
        Tuple of (xgboost_params, lightgbm_params)
    """
    optimizer = GPUOptimizer()
    optimizer.print_summary()

    xgb_params = optimizer.get_xgboost_params()
    lgb_params = optimizer.get_lightgbm_params()

    return xgb_params, lgb_params


if __name__ == '__main__':
    # Test the optimizer
    print("Testing GPU Optimizer...\n")

    optimizer = GPUOptimizer()
    optimizer.print_summary()

    print("\nTesting XGBoost parameter generation:")
    xgb_params = optimizer.get_xgboost_params({
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200
    })

    print("\nTesting LightGBM parameter generation:")
    lgb_params = optimizer.get_lightgbm_params({
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200
    })

    print("\nFull recommendations:")
    recommendations = optimizer.get_training_recommendations()
    print(recommendations)
