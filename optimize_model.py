import os
import pandas as pd
import numpy as np
from pipeline import combine_two_datasets, run_pipeline_oof
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
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
    
    # Load processed data
    flow_data = pd.read_csv("data/processed/flow_accumulation_locations.csv")
    flow_data['flow_accumulation'] = np.log1p(flow_data['flow_accumulation'])
    
    MERGE_KEYS_COORD = ['Latitude', 'Longitude']
    wq_data = combine_two_datasets(wq_data, flow_data, keys=MERGE_KEYS_COORD)
    
    land_data = pd.read_csv("data/processed/land_use_worldcover_1km_locations.csv")
    pt = PowerTransformer(method='box-cox', standardize=True)
    urban = pd.DataFrame(land_data['pct_urban']) + 1
    agri = pd.DataFrame(land_data['pct_agricultural']) + 1
    land_data['pct_agricultural'] = pt.fit_transform(agri)
    land_data['pct_urban'] = pt.fit_transform(urban)
    
    wq_data = combine_two_datasets(wq_data, land_data, keys=MERGE_KEYS_COORD)
    
    selected_features = ['swir22', 'NDMI', 'MNDWI', 'pet', "flow_accumulation", "pct_agricultural", "pct_urban"]
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    X = wq_data[selected_features]
    Y = wq_data[targets]
    
    return X, Y

def optimize_target(X, y, target_name):
    print(f"\n--- Optimizing for {target_name} ---")
    
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
    
    # Hyperparameter grids
    rf_params = {
        'rf__n_estimators': [100, 300, 500],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['sqrt', None]
    }
    
    xgb_params = {
        'xgb__n_estimators': [500, 1000],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__max_depth': [3, 4, 5],
        'xgb__reg_alpha': [0.1, 1, 10],
        'xgb__reg_lambda': [1, 10, 100],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    }
    
    y_log = np.log1p(y)
    
    print("Searching for best RandomForest...")
    rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    rf_search.fit(X, y_log)
    print(f"Best RF R2: {rf_search.best_score_:.4f}")
    
    print("Searching for best XGBoost...")
    xgb_search = RandomizedSearchCV(xgb_pipe, xgb_params, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    xgb_search.fit(X, y_log)
    print(f"Best XGB R2: {xgb_search.best_score_:.4f}")
    
    # Ensemble of best models
    best_rf = rf_search.best_estimator_.named_steps['rf']
    best_xgb = xgb_search.best_estimator_.named_steps['xgb']
    
    ensemble = VotingRegressor([
        ('rf', best_rf),
        ('xgb', best_xgb)
    ])
    
    ensemble_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ensemble', ensemble)
    ])
    
    # Evaluate ensemble with cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X, y_log):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y_log.iloc[train_idx], y_log.iloc[val_idx]
        ensemble_pipe.fit(X_tr, y_tr)
        y_pred = ensemble_pipe.predict(X_va)
        scores.append(r2_score(y_va, y_pred))
    
    ensemble_score = np.mean(scores)
    print(f"Ensemble R2: {ensemble_score:.4f}")
    
    # Save the best model
    if ensemble_score > rf_search.best_score_ and ensemble_score > xgb_search.best_score_:
        best_model = ensemble_pipe
        final_score = ensemble_score
        print("Winning Model: Ensemble")
    elif rf_search.best_score_ > xgb_search.best_score_:
        best_model = rf_search.best_estimator_
        final_score = rf_search.best_score_
        print("Winning Model: RandomForest")
    else:
        best_model = xgb_search.best_estimator_
        final_score = xgb_search.best_score_
        print("Winning Model: XGBoost")
        
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/best_model_{target_name.replace(" ", "_")}.joblib')
    
    return final_score

def main():
    X, Y = load_and_preprocess_data()
    
    final_scores = {}
    for target_col in Y.columns:
        score = optimize_target(X, Y[target_col], target_col)
        final_scores[target_col] = score
        
    print("\n--- Final Summary of Optimized Scores ---")
    for target, score in final_scores.items():
        print(f"{target}: {score:.4f}")

if __name__ == "__main__":
    main()
