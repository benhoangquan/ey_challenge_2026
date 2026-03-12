import os
import pandas as pd
import numpy as np
from pipeline import combine_two_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

def load_and_preprocess_data():
    # Load original datasets
    Water_Quality_df = pd.read_csv('./data/original/water_quality_training_dataset.csv')
    landsat_train_features = pd.read_csv('./data/original/landsat_features_training.csv')
    Terraclimate_df = pd.read_csv('./data/original/terraclimate_features_training.csv')
    
    MERGE_KEYS = ['Latitude', 'Longitude', 'Sample Date']
    wq_data = combine_two_datasets(Water_Quality_df, landsat_train_features, Terraclimate_df, keys=MERGE_KEYS)
    
    # Load physical features
    flow_data = pd.read_csv("data/processed/flow_accumulation_locations.csv")
    flow_data['flow_accumulation'] = np.log1p(flow_data['flow_accumulation'])
    
    elev_data = pd.read_csv("data/processed/elevation_gradient_locations.csv")
    elev_data = elev_data[['Latitude', 'Longitude', 'elevation_m']]
    
    MERGE_KEYS_COORD = ['Latitude', 'Longitude']
    wq_data = combine_two_datasets(wq_data, flow_data, keys=MERGE_KEYS_COORD)
    wq_data = combine_two_datasets(wq_data, elev_data, keys=MERGE_KEYS_COORD)
    
    # Feature Engineering
    wq_data['Sample Date'] = pd.to_datetime(wq_data['Sample Date'], dayfirst=True)
    wq_data['Month'] = wq_data['Sample Date'].dt.month
    
    # Spectral ratio: SWIR22 / Green
    wq_data['swir_green_ratio'] = wq_data['swir22'] / (wq_data['green'] + 1e-6)
    
    # Define groups for GroupKFold
    wq_data['location_group'] = wq_data['Latitude'].astype(str) + "_" + wq_data['Longitude'].astype(str)
    
    selected_features = [
        'swir22', 'NDMI', 'MNDWI', 'pet', 
        'Latitude', 'Longitude', 'Month', 
        'flow_accumulation', 'elevation_m', 'swir_green_ratio'
    ]
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    X = wq_data[selected_features + ['location_group']]
    Y = wq_data[targets]
    
    return X, Y

def optimize_target(X, y, target_name):
    print(f"\n--- Optimizing for {target_name} (v3) ---")
    
    groups = X['location_group']
    X_features = X.drop(columns=['location_group'])
    
    # Define pipelines
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    
    xgb_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor(random_state=42, tree_method='hist'))
    ])
    
    # Hyperparameter grids (Balanced)
    rf_params = {
        'rf__n_estimators': [300],
        'rf__max_depth': [10, 15, 20],
        'rf__min_samples_leaf': [2, 5],
        'rf__max_features': ['sqrt']
    }
    
    xgb_params = {
        'xgb__n_estimators': [500],
        'xgb__learning_rate': [0.01, 0.05],
        'xgb__max_depth': [3, 4, 5],
        'xgb__reg_alpha': [1, 5],
        'xgb__reg_lambda': [5, 50],
        'xgb__subsample': [0.8],
        'xgb__colsample_bytree': [0.8]
    }
    
    # Try log transform for target
    y_log = np.log1p(y)
    
    cv = GroupKFold(n_splits=5)
    
    print("Optimizing RandomForest...")
    rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=5, cv=cv, scoring='r2', n_jobs=-1, random_state=42)
    rf_search.fit(X_features, y_log, groups=groups)
    
    print("Optimizing XGBoost...")
    xgb_search = RandomizedSearchCV(xgb_pipe, xgb_params, n_iter=5, cv=cv, scoring='r2', n_jobs=-1, random_state=42)
    xgb_search.fit(X_features, y_log, groups=groups)
    
    # Ensemble of best models
    best_rf = rf_search.best_estimator_.named_steps['rf']
    best_xgb = xgb_search.best_estimator_.named_steps['xgb']
    
    ensemble = VotingRegressor([
        ('rf', best_rf),
        ('xgb', best_xgb)
    ], weights=[0.6, 0.4])
    
    ensemble_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ensemble', ensemble)
    ])
    
    # Final cross-validation score for ensemble
    scores = []
    for train_idx, val_idx in cv.split(X_features, y_log, groups=groups):
        X_tr, X_va = X_features.iloc[train_idx], X_features.iloc[val_idx]
        y_tr, y_va = y_log.iloc[train_idx], y_log.iloc[val_idx]
        ensemble_pipe.fit(X_tr, y_tr)
        y_pred = ensemble_pipe.predict(X_va)
        scores.append(r2_score(y_va, y_pred))
    
    ensemble_score = np.mean(scores)
    print(f"Ensemble R2 (GroupKFold, Log-y): {ensemble_score:.4f}")
    
    # Refit on all data
    ensemble_pipe.fit(X_features, y_log)
        
    os.makedirs('models', exist_ok=True)
    joblib.dump(ensemble_pipe, f'models/best_model_{target_name.replace(" ", "_")}.joblib')
    
    return ensemble_score

def main():
    X, Y = load_and_preprocess_data()
    
    final_scores = {}
    for target_col in Y.columns:
        score = optimize_target(X, Y[target_col], target_col)
        final_scores[target_col] = score
        
    print("\n--- Final Summary of Optimized Scores (v3 - RF+XGB Ensemble) ---")
    for target, score in final_scores.items():
        print(f"{target}: {score:.4f}")

if __name__ == "__main__":
    main()
