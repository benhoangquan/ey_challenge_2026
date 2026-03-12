import os
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, learning_curve
from sklearn.pipeline import Pipeline

try:
    import ee
except ImportError:  # earthengine-api is optional unless using flow accumulation helpers
    ee = None




def combine_two_datasets(*datasets, keys):
    '''
    Returns a single dataset by merging all inputs on the given keys (inner join).
    Supports two or more datasets.
    '''
    if len(datasets) < 2:
        raise ValueError("At least two datasets required")
    data = reduce(lambda left, right: pd.merge(left, right, on=keys), datasets)
    data = data.loc[:, ~data.columns.duplicated()]
    return data


def analyze_feature_correlation(X: pd.DataFrame, y: pd.Series):
    """Plot feature correlations with target. y must have same index as X."""
    df = pd.merge(X, y.rename('_target'), left_index=True, right_index=True)
    corr_series = df.corr()['_target'].drop('_target', errors='ignore')
    corr_df = corr_series.sort_values(key=lambda s: s.abs(), ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in corr_df.values]
    ax.barh(corr_df.index, corr_df.values, color=colors, edgecolor='white', linewidth=0.7)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Correlation with target', fontsize=11)
    ax.set_title('Feature Correlation Analysis', fontsize=13, fontweight='bold')
    for i, (val, name) in enumerate(zip(corr_df.values, corr_df.index)):
        ax.text(val + 0.01 if val >= 0 else val - 0.01, i, f'{val:.3f}',
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    plt.tight_layout()
    return fig


def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def _make_pipeline(
    kind='full',
    rfe_n_features_to_select=None,
    rfe_step=2,
    xgb_n_estimators=1000,
    xgb_learning_rate=0.1,
    xgb_max_depth=6,
    xgb_reg_lambda=1.0,
    xgb_reg_alpha=0.0,
):
    """Create a modeling pipeline.

    - **'full'**: imputer → scaler → RFE → XGBoost
      - RFE uses an internal XGBoost estimator for feature selection.
      - Final model is an XGBoost regressor with configurable regularization.
    - **'simple'**: imputer → scaler → RandomForest (benchmark-style).
    """
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
    if kind == 'full':
        # Base estimator for RFE (smaller model for speed)
        rfe_estimator = xgb.XGBRegressor(
            n_estimators=min(300, xgb_n_estimators),
            learning_rate=xgb_learning_rate,
            max_depth=xgb_max_depth,
            tree_method='hist',
            reg_lambda=xgb_reg_lambda,
            reg_alpha=xgb_reg_alpha,
        )

        rfe = RFE(
            estimator=rfe_estimator,
            n_features_to_select=rfe_n_features_to_select,
            step=rfe_step,
        )

        # Final XGBoost model (can be larger / more regularized)
        xgb_model = xgb.XGBRegressor(
            n_estimators=xgb_n_estimators,
            learning_rate=xgb_learning_rate,
            max_depth=xgb_max_depth,
            tree_method='hist',
            reg_lambda=xgb_reg_lambda,
            reg_alpha=xgb_reg_alpha,
        )

        steps.extend([
            ('rfe', rfe),
            ('xgb', xgb_model),
        ])
    else:
        steps.append(('model', RandomForestRegressor(n_estimators=100, random_state=42)))
    return Pipeline(steps)


def train_model(
    X_train,
    y_train,
    rfe_n_features_to_select=None,
    rfe_step=1,
    xgb_n_estimators=1000,
    xgb_learning_rate=0.1,
    xgb_max_depth=6,
    xgb_reg_lambda=1.0,
    xgb_reg_alpha=0.0,
):
    """Full pipeline: imputer → scaler → RFE → XGBoost. Fitted only on X_train (no leakage).
    Traces learning curve during training using wandb."""
    # Split training data for validation tracking
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create preprocessing + feature selection steps up to XGBoost
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    rfe_estimator = xgb.XGBRegressor(
        n_estimators=min(300, xgb_n_estimators),
        learning_rate=xgb_learning_rate,
        max_depth=xgb_max_depth,
        tree_method='hist',
        reg_lambda=xgb_reg_lambda,
        reg_alpha=xgb_reg_alpha,
    )
    rfe = RFE(
        estimator=rfe_estimator,
        n_features_to_select=rfe_n_features_to_select,
        step=rfe_step,
    )
    
    # Fit preprocessing steps
    X_tr_imputed = imputer.fit_transform(X_tr)
    X_tr_scaled = scaler.fit_transform(X_tr_imputed)
    X_tr_rfe = rfe.fit_transform(X_tr_scaled, y_tr)
    
    # Transform validation set
    X_val_imputed = imputer.transform(X_val)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_val_rfe = rfe.transform(X_val_scaled)
    
    # Create XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=xgb_n_estimators,
        learning_rate=xgb_learning_rate,
        max_depth=xgb_max_depth,
        tree_method='hist',
        reg_lambda=xgb_reg_lambda,
        reg_alpha=xgb_reg_alpha,
    )
    
    # Store evaluation results for learning curve tracking
    evals_result = {}
    
    # Fit XGBoost with evaluation set (use default eval metric for the objective)
    xgb_model.fit(
        X_tr_rfe, y_tr,
        eval_set=[(X_tr_rfe, y_tr), (X_val_rfe, y_val)],
        verbose=False
    )
    
    # Log learning curve to wandb from evaluation results
    if wandb.run is not None and hasattr(xgb_model, 'evals_result_'):
        evals_result = xgb_model.evals_result_
        # evals_result structure: {'validation_0': {'rmse': [...]}, 'validation_1': {'rmse': [...]}}
        eval_names = list(evals_result.keys())
        for dataset_idx, dataset_name in enumerate(eval_names):
            dataset_label = "train" if dataset_idx == 0 else "val"
            for metric_name, values in evals_result[dataset_name].items():
                # Log each iteration's metric value
                for iteration, value in enumerate(values):
                    metric_key = f"learning_curve/{dataset_label}_{metric_name}"
                    wandb.log({metric_key: value}, step=iteration)
    
    # Reconstruct full pipeline
    model_pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('rfe', rfe),
        ('xgb', xgb_model)
    ])
    
    # Fit on full training set for final model
    model_pipeline.fit(X_train, y_train)

    return model_pipeline


def train_model_simple(X_train, y_train):
    """Benchmark-style pipeline: imputer → scaler → RandomForest. Often better test R² than full (PCA+XGB)."""
    model_pipeline = _make_pipeline('simple')
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

def evaluate_model(model, X_scaled, y_true, dataset_name="Test"):
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{dataset_name} Evaluation:")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    return y_pred, r2, rmse

def run_pipeline(
    X,
    y,
    param_name="Parameter",
    pipeline_kind='full',
    **full_pipeline_kwargs,
):
    """Single train/test split.

    - **pipeline_kind='full'**: imputer → scaler → RFE → XGBoost (with configurable regularization).
      - Extra keyword arguments (only used when `pipeline_kind='full'`):
        - `rfe_n_features_to_select`: int or None – number of features to keep (default: None = keep all).
        - `rfe_step`: int or float – number (or fraction) of features to remove at each step (default: 1).
        - `xgb_n_estimators`: int – boosting rounds (default: 1000).
        - `xgb_learning_rate`: float – learning rate (default: 0.1).
        - `xgb_max_depth`: int – max tree depth (default: 6).
        - `xgb_reg_lambda`: float – L2 regularization term (default: 1.0).
        - `xgb_reg_alpha`: float – L1 regularization term (default: 0.0).
    - **pipeline_kind='simple'**: imputer → scaler → RandomForest (benchmark-style).
    """
    print(f"\n{'='*60}")
    print(f"Training Model for {param_name} [pipeline={pipeline_kind}]")
    print(f"{'='*60}")
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    if pipeline_kind == 'simple':
        model = train_model_simple(X_train, y_train)
    else:
        model = train_model(X_train, y_train, **full_pipeline_kwargs)
    
    # Evaluate (in-sample)
    y_train_pred, r2_train, rmse_train = evaluate_model(model, X_train, y_train, "Train")
    
    # Evaluate (out-sample)
    y_test_pred, r2_test, rmse_test = evaluate_model(model, X_test, y_test, "Test")
    
    # Return summary
    results = {
        "Parameter": param_name,
        "Pipeline": pipeline_kind,
        "R2_Train": r2_train,
        "RMSE_Train": rmse_train,
        "R2_Test": r2_test,
        "RMSE_Test": rmse_test
    }

    # Log to Weights & Biases if a run is active
    if wandb.run is not None:
        wandb.log({
            "param_name": param_name,
            "pipeline_kind": pipeline_kind,
            f"{param_name}/R2_Train": r2_train,
            f"{param_name}/RMSE_Train": rmse_train,
            f"{param_name}/R2_Test": r2_test,
            f"{param_name}/RMSE_Test": rmse_test,
        })

    return model, pd.DataFrame([results])


def run_pipeline_oof(
    X,
    y,
    X_test=None,
    n_splits=5,
    param_name="Parameter",
    pipeline_kind='simple',
    random_state=42,
    **full_pipeline_kwargs,
):
    """
    Out-of-fold validation: train one model per fold, aggregate OOF predictions and (optionally) test predictions.
    pipeline_kind: 'simple' (imputer + scaler + RandomForest, benchmark-style) or 'full' (imputer + scaler + PCA + XGBoost).
    Returns: (models, oof_preds, results_df, test_preds or None)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n = len(X)
    oof_preds = np.zeros(n)
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models = []

    print(f"\n{'='*60}")
    print(f"OOF Training ({n_splits}-Fold) for {param_name} [pipeline={pipeline_kind}]")
    print(f"{'='*60}")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if pipeline_kind == 'simple':
            pipe = _make_pipeline('simple')
        else:
            pipe = _make_pipeline('full', **full_pipeline_kwargs)
        pipe.fit(X_tr, y_tr)

        oof_preds[va_idx] = pipe.predict(X_va)
        if X_test is not None:
            test_preds += pipe.predict(X_test) / n_splits

        models.append(pipe)
        fold_r2 = r2_score(y_va, oof_preds[va_idx])
        fold_rmse = np.sqrt(mean_squared_error(y_va, oof_preds[va_idx]))
        print(f"  Fold {fold} | R²: {fold_r2:.4f} | RMSE: {fold_rmse:.3f}")

        # Per-fold logging
        if wandb.run is not None:
            wandb.log({
                "param_name": param_name,
                "pipeline_kind": pipeline_kind,
                "fold": fold,
                f"{param_name}/Fold_R2": fold_r2,
                f"{param_name}/Fold_RMSE": fold_rmse,
            })

    oof_r2 = r2_score(y, oof_preds)
    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\n  OOF R²: {oof_r2:.4f} | OOF RMSE: {oof_rmse:.3f}")

    results = pd.DataFrame([{
        "Parameter": param_name,
        "Pipeline": pipeline_kind,
        "N_Splits": n_splits,
        "R2_OOF": oof_r2,
        "RMSE_OOF": oof_rmse,
    }])

    # Overall OOF metrics logging
    if wandb.run is not None:
        wandb.log({
            "param_name": param_name,
            "pipeline_kind": pipeline_kind,
            f"{param_name}/R2_OOF": oof_r2,
            f"{param_name}/RMSE_OOF": oof_rmse,
        })

    return models, oof_preds, results, test_preds


# ---------------------------------------------------------------------------
# Flow accumulation data collection (Google Earth Engine / HydroSHEDS)
# ---------------------------------------------------------------------------

def _ensure_ee_initialized():
    """
    Ensure Google Earth Engine is initialized.

    Users must have run `ee.Authenticate()` at least once on the machine.
    """
    if ee is None:
        raise ImportError(
            "earthengine-api is required for flow accumulation helpers. "
            "Install with `pip install earthengine-api`."
        )
    try:
        # Safe to call multiple times; will no-op if already initialized.
        ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Google Earth Engine is not initialized. "
            "Run `ee.Authenticate()` once, then `ee.Initialize()` in your session "
            "before calling flow accumulation helpers."
        ) from exc


def _points_to_feature_collection(locations_df: pd.DataFrame):
    """Build ee.FeatureCollection from DataFrame with Latitude, Longitude."""
    features = []
    for i, row in locations_df.iterrows():
        lon, lat = float(row["Longitude"]), float(row["Latitude"])
        pt = ee.Geometry.Point([lon, lat])
        feat = ee.Feature(pt, {"id": int(i)})
        features.append(feat)
    return ee.FeatureCollection(features)


def _parse_flow_accumulation_result(result: dict) -> pd.DataFrame:
    """Parse `sampleRegions().getInfo()` result into a DataFrame."""
    features = result.get("features", [])
    rows = []
    for f in features:
        props = f.get("properties", {})
        idx = props.get("id")
        acc = props.get("b1")  # band name in WWF/HydroSHEDS/15ACC
        if acc is None:
            acc = np.nan
        rows.append({"id": idx, "flow_accumulation": acc})
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def collect_flow_accumulation_for_locations(
    locations: pd.DataFrame,
    scale_m: int = 463,
) -> pd.DataFrame:
    """
    Attach HydroSHEDS flow accumulation to unique locations.

    Parameters
    ----------
    locations : DataFrame
        Must contain `Latitude` and `Longitude` columns.
    scale_m : int, optional
        Sampling scale in meters (15 arc-second HydroSHEDS ≈ 463 m).

    Returns
    -------
    DataFrame
        Copy of `locations` with an added `flow_accumulation` column.
        Duplicate (Latitude, Longitude) pairs share the same value.
    """
    _ensure_ee_initialized()

    # Unique coordinate pairs to avoid redundant GEE calls
    unique_locs = (
        locations[["Latitude", "Longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    points_fc = _points_to_feature_collection(unique_locs)

    flow_acc_image = ee.Image("WWF/HydroSHEDS/15ACC").select("b1")
    sampled = flow_acc_image.sampleRegions(
        collection=points_fc,
        scale=scale_m,
        geometries=False,
    )

    result = sampled.getInfo()
    acc_df = _parse_flow_accumulation_result(result)

    unique_locs_with_acc = unique_locs.copy()
    unique_locs_with_acc["flow_accumulation"] = acc_df["flow_accumulation"].values

    # Merge back to original locations so duplicates keep their coordinates
    locations_with_acc = locations.merge(
        unique_locs_with_acc,
        on=["Latitude", "Longitude"],
        how="left",
    )
    return locations_with_acc


def generate_submission_flow_accumulation(
    data_root: str = "data",
    template_filename: str = "submission_template.csv",
    output_filename: str = "submission_flow_accumulation_locations.csv",
) -> pd.DataFrame:
    """
    Reproduce the notebook flow-accumulation data collection inside the pipeline.

    - Loads `{data_root}/original/{template_filename}`.
    - Computes HydroSHEDS flow accumulation for unique (Latitude, Longitude) pairs.
    - Saves unique locations with `flow_accumulation` to
      `{data_root}/processed/{output_filename}`.

    Returns
    -------
    DataFrame
        The unique locations dataframe with `flow_accumulation` that was saved.
    """
    original_dir = os.path.join(data_root, "original")
    processed_dir = os.path.join(data_root, "processed")

    in_path = os.path.join(original_dir, template_filename)
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Could not find submission template at '{in_path}'. "
            "Adjust `data_root` or `template_filename`."
        )

    df = pd.read_csv(in_path)
    locations = df[["Latitude", "Longitude"]].drop_duplicates().reset_index(drop=True)

    locations_with_acc = collect_flow_accumulation_for_locations(locations)

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, output_filename)
    locations_with_acc.to_csv(out_path, index=False)

    print(f"Saved {len(locations_with_acc)} rows with flow_accumulation to {out_path}")
    return locations_with_acc