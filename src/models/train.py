"""
=========================================================
Model Training Module
=========================================================

Train machine learning models for
Store Sales Forecasting.

Models
------
- XGBoost
- LightGBM

Author : Shariq Zia
Project: Store Sales Forecasting
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import MODEL_DIR


# =========================================================
# FEATURE PREPARATION
# =========================================================

def prepare_features(
    train: pd.DataFrame,
    target: str = "log_sales"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare model features and target.
    """

    df = train.copy()

    drop_columns = [

        "id",

        "date",

        "sales",

        "log_sales"

    ]

    X = df.drop(
        columns=[
            col
            for col in drop_columns
            if col in df.columns
        ]
    )

    y = df[target]

    return X, y


# =========================================================
# CATEGORY ENCODING
# =========================================================

def encode_categories(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame
):

    """
    Encode categorical variables using
    LabelEncoder.
    """

    X_train = X_train.copy()
    X_valid = X_valid.copy()

    encoders = {}

    categorical = X_train.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical:

        encoder = LabelEncoder()

        combined = pd.concat(
            [
                X_train[col],
                X_valid[col]
            ]
        ).astype(str)

        encoder.fit(
            combined
        )

        X_train[col] = encoder.transform(
            X_train[col].astype(str)
        )

        X_valid[col] = encoder.transform(
            X_valid[col].astype(str)
        )

        encoders[col] = encoder

    return X_train, X_valid, encoders


# =========================================================
# TIME VALIDATION SPLIT
# =========================================================

def time_split(
    train: pd.DataFrame,
    validation_days: int = 16
):

    """
    Split using the last N days
    for validation.
    """

    df = train.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    cutoff = (

        df["date"].max()

        - pd.Timedelta(
            days=validation_days
        )

    )

    train_df = df[
        df["date"] <= cutoff
    ]

    valid_df = df[
        df["date"] > cutoff
    ]

    return train_df, valid_df


# =========================================================
# XGBOOST
# =========================================================

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series
):

    """
    Train XGBoost model.
    """

    model = XGBRegressor(

        objective="reg:squarederror",

        n_estimators=1000,

        learning_rate=0.05,

        max_depth=8,

        subsample=0.8,

        colsample_bytree=0.8,

        tree_method="hist",

        random_state=42,

        n_jobs=-1

    )

    model.fit(

        X_train,

        y_train

    )

    return model


# =========================================================
# LIGHTGBM
# =========================================================

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series
):

    """
    Train LightGBM model.
    """

    model = LGBMRegressor(

        n_estimators=1000,

        learning_rate=0.05,

        num_leaves=64,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        n_jobs=-1

    )

    model.fit(

        X_train,

        y_train

    )

    return model


# =========================================================
# SAVE ARTIFACT
# =========================================================

def save_artifact(
    artifact,
    filename: str
) -> None:
    """
    Save any model artifact.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        artifact,
        MODEL_DIR / filename
    )


# =========================================================
# SAVE TRAINING METADATA
# =========================================================

def save_training_metadata(
    X_train: pd.DataFrame,
    y_train: pd.Series
):
    """
    Save training metadata.
    """

    metadata = {

        "training_date":
            datetime.now(),

        "rows":
            len(X_train),

        "feature_count":
            len(X_train.columns),

        "features":
            X_train.columns.tolist(),

        "target":
            "log_sales",

        "models":
            [
                "XGBoost",
                "LightGBM"
            ],

        "validation_strategy":
            "Last 16 Days"

    }

    save_artifact(
        metadata,
        "training_metadata.pkl"
    )

# =========================================================
# MASTER TRAIN FUNCTION
# =========================================================

def train_models(
    train_features: pd.DataFrame
) -> Dict:
    """
    Complete model training pipeline.

    Steps
    -----
    1. Time-based train/validation split
    2. Feature preparation
    3. Encode categorical variables
    4. Train XGBoost
    5. Train LightGBM
    6. Save all artifacts
    """

    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    print("\nPreparing training and validation data...")

    # -----------------------------------
    # Train / Validation Split
    # -----------------------------------

    train_df, valid_df = time_split(
        train_features
    )

    # -----------------------------------
    # Prepare Features
    # -----------------------------------

    X_train, y_train = prepare_features(
        train_df
    )

    X_valid, y_valid = prepare_features(
        valid_df
    )

    # -----------------------------------
    # Encode Categories
    # -----------------------------------

    X_train, X_valid, encoders = encode_categories(
        X_train,
        X_valid
    )

    # -----------------------------------
    # Train Models
    # -----------------------------------

    print("\nTraining XGBoost...")

    xgb_model = train_xgboost(
        X_train,
        y_train
    )

    print("✓ XGBoost completed")

    print("\nTraining LightGBM...")

    lgb_model = train_lightgbm(
        X_train,
        y_train
    )

    print("✓ LightGBM completed")

    # -----------------------------------
    # Save Artifacts
    # -----------------------------------

    print("\nSaving model artifacts...")

    save_artifact(
        xgb_model,
        "xgboost_model.pkl"
    )

    save_artifact(
        lgb_model,
        "lightgbm_model.pkl"
    )

    save_artifact(
        X_train.columns.tolist(),
        "feature_columns.pkl"
    )

    save_artifact(
        encoders,
        "category_encoders.pkl"
    )

    save_training_metadata(
        X_train,
        y_train
    )

    print("✓ Artifacts saved")

    # -----------------------------------
    # Training Summary
    # -----------------------------------

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    print(f"Training rows     : {len(X_train):,}")
    print(f"Validation rows   : {len(X_valid):,}")
    print(f"Features          : {X_train.shape[1]}")
    print(f"XGBoost Model     : Saved")
    print(f"LightGBM Model    : Saved")
    print(f"Feature Columns   : Saved")
    print(f"Encoders          : Saved")
    print(f"Metadata          : Saved")

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETED")
    print("=" * 70)

    return {

        "xgboost": xgb_model,

        "lightgbm": lgb_model,

        "X_train": X_train,

        "y_train": y_train,

        "X_valid": X_valid,

        "y_valid": y_valid,

        "feature_columns": X_train.columns.tolist(),

        "encoders": encoders

    }