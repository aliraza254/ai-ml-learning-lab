"""
03_pandas_data_wrangling.py
-----------------------------------------------------------------------------
Hands-On Data Wrangling, Cleaning & Feature Preparation with Pandas.

Why Pandas Matters:
Machine learning models cannot take raw, messy database tables directly.
Raw real-world data contains null values, unencoded strings, and skewness.
Pandas transforms raw messy tables into clean numerical feature matrices (X, y).

Key Topics:
1. Simulating a Realistic Raw Dataset (with nulls, categories, edge cases)
2. Structural Inspection & Summary Statistics
3. Handling Missing Data (Imputation strategy)
4. Feature Engineering: Binning, Flagging & One-Hot Encoding
5. Separation into Clean Feature Matrix X and Target Vector y
-----------------------------------------------------------------------------
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding for cross-platform terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_raw_messy_data(n_samples: int = 1000) -> pd.DataFrame:
    """Simulates a realistic, raw customer dataset with messy real-world artifacts."""
    np.random.seed(42)

    data = {
        "customer_id": [f"USR_{i:04d}" for i in range(n_samples)],
        "age": np.random.choice([22, 28, 35, 42, 50, 65, np.nan], size=n_samples, p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05]),
        "subscription_plan": np.random.choice(["Basic", "Standard", "Premium", np.nan], size=n_samples, p=[0.45, 0.35, 0.15, 0.05]),
        "monthly_charges": np.random.uniform(19.99, 120.0, size=n_samples),
        "total_charges": np.random.uniform(50.0, 3500.0, size=n_samples),
        "support_tickets": np.random.poisson(lam=1.5, size=n_samples),
        "churn": np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]),
    }

    df = pd.DataFrame(data)

    # Introduce some artificial noisy edge cases (e.g., negative charges or extreme outliers)
    df.loc[10, "monthly_charges"] = -99.0  # corrupted entry
    return df


def clean_and_prepare_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Production-style data cleaning pipeline.
    Returns:
        X (features dataframe)
        y (target series)
    """
    df = raw_df.copy()

    # 1. Sanity Filtering: Remove negative/corrupted numerical values
    df = df[df["monthly_charges"] > 0].copy()

    # 2. Missing Value Imputation
    # Numeric: Fill with median (robust against outliers)
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)

    # Categorical: Fill with mode (most frequent value)
    mode_plan = df["subscription_plan"].mode()[0]
    df["subscription_plan"] = df["subscription_plan"].fillna(mode_plan)

    # 3. Feature Engineering: Create domain-specific indicators
    df["high_support_risk"] = (df["support_tickets"] >= 4).astype(int)
    df["charge_per_age"] = (df["monthly_charges"] / df["age"]).round(3)

    # 4. One-Hot Encoding for categorical features
    df = pd.get_dummies(df, columns=["subscription_plan"], prefix="plan", drop_first=True, dtype=int)

    # 5. Drop identification columns that have no predictive value
    df = df.drop(columns=["customer_id"])

    # 6. Separate Features (X) and Target (y)
    y = df["churn"]
    X = df.drop(columns=["churn"])

    return X, y


def main():
    print("=" * 65)
    print(">> MODULE 03: PANDAS DATA WRANGLING & FEATURE PIPELINE")
    print("=" * 65)

    print("\n[Step 1] Ingesting simulated raw customer dataset...")
    raw_df = generate_raw_messy_data(n_samples=1_000)
    print(f"  - Raw Shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(f"  - Missing Values Found:\n{raw_df.isnull().sum()[raw_df.isnull().sum() > 0]}")

    print("\n[Step 2] Executing Cleaning & Feature Transformation Pipeline...")
    X, y = clean_and_prepare_features(raw_df)

    print(f"  - Cleaned Feature Matrix (X) Shape: {X.shape}")
    print(f"  - Target Vector (y) Shape: {y.shape}")
    print(f"  - Target Class Distribution (Churn):\n{y.value_counts(normalize=True).round(3).to_dict()}")

    print("\n[Step 3] Inspecting Top 5 Processed Feature Rows:")
    print(X.head())

    # Export cleaned dataset to local processed folder
    output_dir = Path("D:/My Company/ml/ai-ml-learning-lab/01_python_and_math_foundations/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_csv_path = output_dir / "clean_customer_features.csv"
    
    # Save combined cleaned table
    processed_df = X.copy()
    processed_df["churn"] = y
    processed_df.to_csv(clean_csv_path, index=False)
    print(f"\n[OK] Cleaned feature dataset saved to: {clean_csv_path}")

    print("=" * 65)
    print(">> Lesson 3 complete! You have a clean feature pipeline ready for Scikit-Learn.")


if __name__ == "__main__":
    main()
