import os
import pandas as pd
import numpy as np
from pipeline import combine_two_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

def calculate_indices(df):
    # NDSI: (Green - SWIR16) / (Green + SWIR16)
    df['NDSI'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + 1e-6)
    # NDWI: (Green - NIR) / (Green + NIR)
    df['NDWI'] = (df['green'] - df['nir']) / (df['green'] + df['nir'] + 1e-6)
    # SWIR Ratio: SWIR22 / SWIR16
    df['swir_ratio'] = df['swir22'] / (df['swir16'] + 1e-6)
    return df

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
    
    MERGE_KEYS_COORD = ['Latitude', 'Longitude']
    wq_data = combine_two_datasets(wq_data, flow_data, keys=MERGE_KEYS_COORD)
    
    # Calculate New Indices
    wq_data = calculate_indices(wq_data)
    
    # Feature Engineering
    wq_data['Sample Date'] = pd.to_datetime(wq_data['Sample Date'], dayfirst=True)
    wq_data['Month'] = wq_data['Sample Date'].dt.month
    
    # Define groups for GroupKFold
    wq_data['location_group'] = wq_data['Latitude'].astype(str) + "_" + wq_data['Longitude'].astype(str)
    
    selected_features = [
        'swir22', 'NDMI', 'MNDWI', 'pet', 
        'Latitude', 'Longitude', 'Month', 
        'flow_accumulation', 'NDSI', 'NDWI', 'swir_ratio'
    ]
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    X = wq_data[selected_features + ['location_group']]
    Y = wq_data[targets]
    
    return X, Y

def optimize_target(X, y, target_name):
    print(f"\n--- Optimizing for {target_name} (v4 - RF) ---")
    
    groups = X['location_group']
    X_features = X.drop(columns=['location_group'])
    
    # Define pipeline
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    
    # Hyperparameter grid (Standard for RF)
    rf_params = {
        'rf__n_estimators': [300],
        'rf__max_depth': [10, 15, None],
        'rf__min_samples_leaf': [2, 5],
        'rf__max_features': ['sqrt']
    }
    
    # NO LOG TRANSFORM on target
    
    cv = GroupKFold(n_splits=5)
    
    print("Optimizing RandomForest...")
    rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=5, cv=cv, scoring='r2', n_jobs=-1, random_state=42)
    rf_search.fit(X_features, y, groups=groups)
    
    best_model = rf_search.best_estimator_
    final_score = rf_search.best_score_
    print(f"Best RF R2 (GroupKFold, No Log): {final_score:.4f}")
    
    # Refit on all data
    best_model.fit(X_features, y)
        
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/best_model_{target_name.replace(" ", "_")}.joblib')
    
    return final_score

def main():
    X, Y = load_and_preprocess_data()
    
    final_scores = {}
    for target_col in Y.columns:
        score = optimize_target(X, Y[target_col], target_col)
        final_scores[target_col] = score
        
    print("\n--- Final Summary of Optimized Scores (v4 - RandomForest) ---")
    for target, score in final_scores.items():
        print(f"{target}: {score:.4f}")

if __name__ == "__main__":
    main()
