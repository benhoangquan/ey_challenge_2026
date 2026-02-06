import pandas as pd
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import wandb


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

def _make_pipeline(kind='full'):
    """Kind: 'full' = imputer + scaler + PCA + XGBoost (current). 'simple' = imputer + scaler + RandomForest (benchmark-style)."""
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
    if kind == 'full':
        steps.extend([
            ('pca', PCA(n_components=0.95)),
            ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, tree_method='hist')),
        ])
    else:
        steps.append(('model', RandomForestRegressor(n_estimators=100, random_state=42)))
    return Pipeline(steps)


def train_model(X_train, y_train):
    """Full pipeline: imputer → scaler → PCA → XGBoost. Fitted only on X_train (no leakage)."""
    model_pipeline = _make_pipeline('full')
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

def run_pipeline(X, y, param_name="Parameter", pipeline_kind='full'):
    """Single train/test split. pipeline_kind: 'full' (PCA+XGB) or 'simple' (RandomForest, benchmark-style)."""
    print(f"\n{'='*60}")
    print(f"Training Model for {param_name} [pipeline={pipeline_kind}]")
    print(f"{'='*60}")
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    train_fn = train_model_simple if pipeline_kind == 'simple' else train_model
    model = train_fn(X_train, y_train)
    
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
    X, y,
    X_test=None,
    n_splits=5,
    param_name="Parameter",
    pipeline_kind='simple',
    random_state=42,
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

        pipe = _make_pipeline(pipeline_kind)
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