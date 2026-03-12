import os
import pandas as pd
import numpy as np
from pipeline import combine_two_datasets, run_pipeline_oof
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
    
    # Define groups for GroupKFold
    wq_data['location_group'] = wq_data['Latitude'].astype(str) + "_" + wq_data['Longitude'].astype(str)
    
    # Feature Engineering: Extract Month
    wq_data['Sample Date'] = pd.to_datetime(wq_data['Sample Date'], dayfirst=True)
    wq_data['Month'] = wq_data['Sample Date'].dt.month
    
    # Match Benchmark features + Lat/Lon + Month
    selected_features = ['swir22', 'NDMI', 'MNDWI', 'pet', 'Latitude', 'Longitude', 'Month']
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    X = wq_data[selected_features + ['location_group']]
    Y = wq_data[targets]
    
    return X, Y

def optimize_target(X, y, target_name):
    print(f"\n--- Optimizing for {target_name} ---")
    
    groups = X['location_group']
    X_features = X.drop(columns=['location_group'])
    
    # Define pipelines
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    
    # Hyperparameter grids (Regularized)
    rf_params = {
        'rf__n_estimators': [300],
        'rf__max_depth': [10, 15, None],
        'rf__min_samples_leaf': [2, 5],
        'rf__max_features': ['sqrt']
    }
    
    cv = GroupKFold(n_splits=5)
    
    print("Searching for best RandomForest...")
    rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=5, cv=cv, scoring='r2', n_jobs=-1, random_state=42)
    rf_search.fit(X_features, y, groups=groups)
    print(f"Best RF R2: {rf_search.best_score_:.4f}")
    
    best_model = rf_search.best_estimator_
    final_score = rf_search.best_score_
        
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/best_model_{target_name.replace(" ", "_")}.joblib')
    
    return final_score

def main():
    X, Y = load_and_preprocess_data()
    
    final_scores = {}
    for target_col in Y.columns:
        score = optimize_target(X, Y[target_col], target_col)
        final_scores[target_col] = score
        
    print("\n--- Final Summary of Optimized Scores (GroupKFold, Benchmark+LatLon) ---")
    for target, score in final_scores.items():
        print(f"{target}: {score:.4f}")

if __name__ == "__main__":
    main()
