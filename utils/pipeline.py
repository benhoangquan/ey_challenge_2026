import pandas as pd
from sklearn.preprocessing import StandardScaler, SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib as plt


def combine_two_datasets(*datasets, keys):
    '''
    Returns a  vertically concatenated dataset.
    Attributes:
    dataset1 - Dataset 1 to be combined 
    dataset2 - Dataset 2 to be combined
    '''
    
    data = pd.merge(datasets, on=keys)
    data = data.loc[:, ~data.columns.duplicated()]
    return data

def analyze_feature_correlation(X: pd.DataFrame, y: pd.DataFrame): 
    df = pd.merge(X, y)
    corr_df = df.corr()[y].drop(y).sort_values(key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in corr_df.values]
    bars = ax.barh(corr_df.index, corr_df.values, color=colors, edgecolor='white', linewidth=0.7)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Correlation with Heart Disease', fontsize=11)
    ax.set_title('Feature Correlation Analysis', fontsize=13, fontweight='bold')
    for i, (val, name) in enumerate(zip(corr_df.values, corr_df.index)):
        ax.text(val + 0.01 if val >= 0 else val - 0.01, i, f'{val:.3f}', 
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    plt.tight_layout()


def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    # This pipeline handles Scaling AND PCA AND XGBoost correctly
    # It will fit ONLY on X_train, preventing leakage
    model_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('xgb', xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            tree_method='hist'
        ))
    ])
    
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

def run_pipeline(X, y, param_name="Parameter"):
    print(f"\n{'='*60}")
    print(f"Training Model for {param_name}")
    print(f"{'='*60}")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate (in-sample)
    y_train_pred, r2_train, rmse_train = evaluate_model(model, X_train, y_train, "Train")
    
    # Evaluate (out-sample)
    y_test_pred, r2_test, rmse_test = evaluate_model(model, X_test, y_test, "Test")
    
    # Return summary
    results = {
        "Parameter": param_name,
        "R2_Train": r2_train,
        "RMSE_Train": rmse_train,
        "R2_Test": r2_test,
        "RMSE_Test": rmse_test
    }
    return model, pd.DataFrame([results])