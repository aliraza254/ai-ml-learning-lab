"""
02_numpy_vectorization.py
-----------------------------------------------------------------------------
Hands-On NumPy Vectorization & Core ML Mathematical Foundations.

Why Vectorization Matters:
Python loops run interpreted bytecode per iteration. NumPy delegates array
operations to highly optimized compiled C/BLAS/LAPACK routines, running 50x-100x faster.

Key Topics:
1. Benchmark: Pure Python Loop vs. Vectorized NumPy (1 Million Elements)
2. Array Broadcasting & Matrix Transformations
3. Implementing Essential ML Loss Functions From Scratch:
   - Mean Squared Error (Regression)
   - Sigmoid & Binary Cross-Entropy Loss (Classification)
   - Z-score Standardization (Feature Scaling)
-----------------------------------------------------------------------------
"""

import sys
import time
import numpy as np

# Ensure UTF-8 output encoding for cross-platform terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Benchmark: Loop vs Vectorization
# ---------------------------------------------------------------------------
def benchmark_loop_vs_vectorized():
    n_elements = 1_000_000
    print(f"\n[Benchmark] Computing Squared Errors across {n_elements:,} predictions...")

    # Python Lists
    y_true_list = [float(i % 100) for i in range(n_elements)]
    y_pred_list = [float((i + 1) % 100) for i in range(n_elements)]

    # NumPy Arrays
    y_true_np = np.array(y_true_list, dtype=np.float32)
    y_pred_np = np.array(y_pred_list, dtype=np.float32)

    # 1. Pure Python Loop
    start = time.perf_counter()
    squared_errors_list = [(yt - yp) ** 2 for yt, yp in zip(y_true_list, y_pred_list)]
    sum_errors_list = sum(squared_errors_list)
    py_time = (time.perf_counter() - start) * 1000

    # 2. Vectorized NumPy
    start = time.perf_counter()
    squared_errors_np = (y_true_np - y_pred_np) ** 2
    sum_errors_np = np.sum(squared_errors_np)
    np_time = (time.perf_counter() - start) * 1000

    speedup = py_time / np_time if np_time > 0 else 0
    print(f"  - Pure Python Loop : {py_time:.2f} ms")
    print(f"  - NumPy Vectorized : {np_time:.2f} ms")
    print(f"  -> Vectorized NumPy is {speedup:.1f}x FASTER!")


# ---------------------------------------------------------------------------
# 2. Essential ML Math Functions from Scratch
# ---------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable Sigmoid activation function: maps (-inf, inf) -> (0, 1)."""
    # Prevent overflow for large negative numbers
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes MSE: (1/N) * sum((y_true - y_pred)^2)."""
    return float(np.mean((y_true - y_pred) ** 2))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Computes Binary Cross-Entropy Loss with epsilon clipping to avoid log(0).
    Loss = - (1/N) * sum( y*log(p) + (1-y)*log(1-p) )
    """
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    return float(loss)


def standard_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    z = (X - mu) / sigma
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Prevent division by zero if a feature has zero variance
    std_safe = np.where(std == 0, 1.0, std)
    X_scaled = (X - mean) / std_safe
    return X_scaled, mean, std


# ---------------------------------------------------------------------------
# 3. Main Demonstration
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(">> MODULE 02: NUMPY VECTORIZATION & ML MATH FOUNDATIONS")
    print("=" * 65)

    # Run Benchmark
    benchmark_loop_vs_vectorized()

    print("\n[ML Math from Scratch]")
    # Simulated true binary labels and raw model logits
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.float32)
    raw_logits = np.array([2.5, -1.8, 1.2, 0.4, -2.1, -0.2, 3.1, -1.5], dtype=np.float32)

    # Sigmoid to convert logits into predicted probabilities
    y_prob = sigmoid(raw_logits)
    bce_loss = binary_cross_entropy(y_true, y_prob)

    print(f"  - Ground Truth Labels : {y_true.tolist()}")
    print(f"  - Model Probabilities : {[round(p, 3) for p in y_prob.tolist()]}")
    print(f"  - Binary Cross-Entropy Loss: {bce_loss:.4f}")

    # Feature Scaling Demonstration
    print("\n[Feature Standardization (Broadcasting in Action)]")
    raw_features = np.array([
        [25.0, 50000.0],  # [Age, Salary]
        [45.0, 120000.0],
        [32.0, 75000.0],
        [58.0, 110000.0]
    ], dtype=np.float32)

    scaled_X, mean, std = standard_scaler(raw_features)
    print(f"  - Feature Means : {mean.tolist()}")
    print(f"  - Feature Stds  : {std.tolist()}")
    print(f"  - Scaled Feature Matrix (Mean~0, Std~1):\n{np.round(scaled_X, 3)}")

    print("\n" + "=" * 65)
    print(">> Lesson 2 complete! You understand vectorization, broadcasting, and loss math.")


if __name__ == "__main__":
    main()
