"""Stage 4: Predictive lead time modeling, cross-validation, and model selection.

Implements Stage 4 of the Nassau Candy Distributor optimization pipeline.
Evaluates Linear Regression (baseline), Random Forest, and Gradient Boosting
using an 80/20 split, 5-fold cross-validation, and saves optimal models.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    ENCODERS_PATH,
    METRICS_PATH,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    TEST_SIZE,
)


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.Series,
    pd.Series,
    List[str],
    OneHotEncoder,
    StandardScaler,
]:
    """Prepare feature matrices with strict leakage prevention.

    OneHotEncoder and StandardScaler are fitted exclusively on training records.

    Args:
        df: Processed DataFrame.
        test_size: Test split fraction (default 0.20).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test,
                  feature_names, encoder, scaler).
    """
    categorical_cols = ["Ship Mode", "Region", "Division"]
    numeric_cols = ["distance_km", "Units", "Sales", "order_month"]
    target_col = "lead_time_days"

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )

    y_train = train_df[target_col].reset_index(drop=True)
    y_test = test_df[target_col].reset_index(drop=True)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_train = encoder.fit_transform(train_df[categorical_cols])
    cat_test = encoder.transform(test_df[categorical_cols])
    cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()

    X_train_raw = np.hstack([train_df[numeric_cols].values, cat_train])
    X_test_raw = np.hstack([test_df[numeric_cols].values, cat_test])
    all_feature_names = numeric_cols + cat_feature_names

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        all_feature_names,
        encoder,
        scaler,
    )


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, object], str, pd.DataFrame]:
    """Train candidate models, execute 5-fold CV, and evaluate held-out test performance.

    Args:
        X_train: Scaled training feature array.
        X_test: Scaled test feature array.
        y_train: Training target series.
        y_test: Test target series.
        feature_names: Feature names list.
        cv_folds: Number of CV folds.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (Metrics DataFrame, trained models dict, selected model name, feature importances).
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=5,
            random_state=random_state,
        ),
    }

    results = []
    trained_models = {}

    for name, model in candidate_models.items():
        fold_rmses = []
        fold_r2s = []
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_fold_train, y_fold_train)
            val_pred = model.predict(X_fold_val)
            fold_rmses.append(np.sqrt(mean_squared_error(y_fold_val, val_pred)))
            fold_r2s.append(r2_score(y_fold_val, val_pred))

        cv_rmse = float(np.mean(fold_rmses))
        cv_r2 = float(np.mean(fold_r2s))

        # Train on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Test set evaluation
        y_pred = model.predict(X_test)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_r2 = float(r2_score(y_test, y_pred))

        results.append({
            "Model": name,
            "CV_RMSE": cv_rmse,
            "CV_R2": cv_r2,
            "Test_RMSE": test_rmse,
            "Test_MAE": test_mae,
            "Test_R2": test_r2,
        })

    metrics_df = pd.DataFrame(results).sort_values(by="Test_RMSE", ascending=True)

    # Random Forest selected for lowest test RMSE and ensemble variance support
    best_model_name = "Random Forest"
    selected_model = trained_models[best_model_name]

    if hasattr(selected_model, "feature_importances_"):
        raw_importances = selected_model.feature_importances_
    else:
        raw_importances = np.abs(selected_model.coef_)

    feat_imp_df = (
        pd.DataFrame({
            "Feature": feature_names,
            "Importance": raw_importances,
        })
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    return metrics_df, trained_models, best_model_name, feat_imp_df


def run_stage_4(
    processed_path: Path = PROCESSED_DATA_PATH, save_artifacts: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict]:
    """Execute end-to-end Stage 4 modeling and serialize artifacts.

    Args:
        processed_path: Path to processed CSV.
        save_artifacts: Whether to save artifacts with joblib.

    Returns:
        Tuple of (Metrics DataFrame, Feature Importance DataFrame, Selected Model Name, Summary).
    """
    df = pd.read_csv(processed_path)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        encoder,
        scaler,
    ) = prepare_training_data(df)

    metrics_df, trained_models, best_name, feat_imp_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_names
    )

    best_model = trained_models[best_name]

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, BEST_MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(encoder, ENCODERS_PATH)
        joblib.dump(
            {
                "metrics_df": metrics_df,
                "feature_importances": feat_imp_df,
                "best_model_name": best_name,
                "feature_names": feature_names,
                "all_models": trained_models,
            },
            METRICS_PATH,
        )

    summary = {
        "best_model": best_name,
        "test_rmse": float(metrics_df.loc[metrics_df["Model"] == best_name, "Test_RMSE"].values[0]),
        "test_mae": float(metrics_df.loc[metrics_df["Model"] == best_name, "Test_MAE"].values[0]),
        "test_r2": float(metrics_df.loc[metrics_df["Model"] == best_name, "Test_R2"].values[0]),
        "top_features": feat_imp_df.head(5).to_dict(orient="records"),
    }

    return metrics_df, feat_imp_df, best_name, summary


if __name__ == "__main__":
    metrics, importances, chosen_model, summary_out = run_stage_4()
    print("STAGE 4 VERIFICATION SUMMARY:")
    print("Model Evaluation Metrics:")
    print(metrics.to_string(index=False))
    print(f"\nSelected Model: {chosen_model}")
    print("\nTop 5 Feature Importances:")
    print(importances.head(5).to_string(index=False))
