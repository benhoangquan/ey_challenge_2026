import os
import pandas as pd
import numpy as np
from pipeline import combine_two_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
from scipy.spatial import cKDTree

def load_data():
    # Training Data
    Water_Quality_df = pd.read_csv('./data/original/water_quality_training_dataset.csv')
    landsat_train_features = pd.read_csv('./data/original/landsat_features_training.csv')
    Terraclimate_df = pd.read_csv('./data/original/terraclimate_features_training.csv')
    
    MERGE_KEYS = ['Latitude', 'Longitude', 'Sample Date']
    wq_train = combine_two_datasets(Water_Quality_df, landsat_train_features, Terraclimate_df, keys=MERGE_KEYS)
    
    flow_train = pd.read_csv("data/processed/flow_accumulation_locations.csv")
    elev_train = pd.read_csv("data/processed/elevation_gradient_locations.csv")[['Latitude', 'Longitude', 'elevation_m']]
    
    MERGE_KEYS_COORD = ['Latitude', 'Longitude']
    wq_train = combine_two_datasets(wq_train, flow_train, keys=MERGE_KEYS_COORD)
    wq_train = combine_two_datasets(wq_train, elev_train, keys=MERGE_KEYS_COORD)
    
    # Validation/Test Data
    landsat_val_features = pd.read_csv('./data/original/landsat_features_validation.csv')
    terraclimate_val_features = pd.read_csv('./data/original/terraclimate_features_validation.csv')
    submission_template = pd.read_csv('./data/original/submission_template.csv')
    
    wq_test = combine_two_datasets(landsat_val_features, terraclimate_val_features, keys=MERGE_KEYS)
    
    flow_test = pd.read_csv("data/processed/submission_flow_accumulation_locations.csv")
    wq_test = combine_two_datasets(wq_test, flow_test, keys=MERGE_KEYS_COORD)
    
    # Spatial interpolation for Elevation (nearest neighbor from train locations)
    train_locs = elev_train[['Latitude', 'Longitude']].values
    test_locs = wq_test[['Latitude', 'Longitude']].values
    tree = cKDTree(train_locs)
    _, idx = tree.query(test_locs, k=1)
    wq_test['elevation_m'] = elev_train.iloc[idx]['elevation_m'].values
    
    return wq_train, wq_test, submission_template

def preprocess_and_predict(wq_train, wq_test):
    selected_features = [
        'swir22', 'NDMI', 'MNDWI', 'pet', 
        'Latitude', 'Longitude', 'Month', 
        'flow_accumulation', 'elevation_m', 'swir_green_ratio'
    ]
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Preprocess training
    wq_train['flow_accumulation'] = np.log1p(wq_train['flow_accumulation'])
    wq_train['Sample Date'] = pd.to_datetime(wq_train['Sample Date'], dayfirst=True)
    wq_train['Month'] = wq_train['Sample Date'].dt.month
    wq_train['swir_green_ratio'] = wq_train['swir22'] / (wq_train['green'] + 1e-6)
    
    # Preprocess test
    wq_test['flow_accumulation'] = np.log1p(wq_test['flow_accumulation'])
    wq_test['Sample Date'] = pd.to_datetime(wq_test['Sample Date'], dayfirst=True)
    wq_test['Month'] = wq_test['Sample Date'].dt.month
    wq_test['swir_green_ratio'] = wq_test['swir22'] / (wq_test['green'] + 1e-6)
    
    # NaN handling for test (ffill/bfill)
    wq_test = wq_test.sort_values(['Latitude', 'Longitude', 'Sample Date'])
    fill_cols = ['swir22', 'NDMI', 'MNDWI', 'swir_green_ratio']
    wq_test[fill_cols] = wq_test.groupby(['Latitude', 'Longitude'])[fill_cols].ffill().bfill()
    
    X_test = wq_test[selected_features]
    predictions = {}
    
    for target in targets:
        print(f"Predicting for {target}...")
        model_path = f'models/best_model_{target.replace(" ", "_")}.joblib'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred_log = model.predict(X_test)
            predictions[target] = np.expm1(y_pred_log) # INVERSE LOG
        else:
            print(f"Warning: Model for {target} not found!")
            predictions[target] = np.zeros(len(X_test)) + wq_train[target].median()
            
    return predictions, wq_test

def main():
    wq_train, wq_test, submission_template = load_data()
    predictions, wq_test_processed = preprocess_and_predict(wq_train, wq_test)
    
    results_df = wq_test_processed[['Latitude', 'Longitude', 'Sample Date']].copy()
    for target, preds in predictions.items():
        results_df[target] = preds
        
    results_df['Sample Date'] = results_df['Sample Date'].dt.strftime('%d-%m-%Y')
    
    final_sub = submission_template[['Latitude', 'Longitude', 'Sample Date']].merge(
        results_df, on=['Latitude', 'Longitude', 'Sample Date'], how='left'
    )
    
    final_sub = final_sub[['Latitude', 'Longitude', 'Sample Date', 'Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']]
    
    for target in ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']:
        median_val = wq_train[target].median()
        final_sub[target] = final_sub[target].fillna(median_val)
        final_sub[target] = final_sub[target].clip(lower=wq_train[target].min(), upper=wq_train[target].max())
    
    final_sub.to_csv('submission.csv', index=False)
    print("Saved submission to submission.csv")

if __name__ == "__main__":
    main()
