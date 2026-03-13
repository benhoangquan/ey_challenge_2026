import os
import pandas as pd
import numpy as np
from pipeline import combine_two_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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

def load_data():
    # Training Data
    Water_Quality_df = pd.read_csv('./data/original/water_quality_training_dataset.csv')
    landsat_train_features = pd.read_csv('./data/original/landsat_features_training.csv')
    Terraclimate_df = pd.read_csv('./data/original/terraclimate_features_training.csv')
    
    MERGE_KEYS = ['Latitude', 'Longitude', 'Sample Date']
    wq_train = combine_two_datasets(Water_Quality_df, landsat_train_features, Terraclimate_df, keys=MERGE_KEYS)
    
    flow_train = pd.read_csv("data/processed/flow_accumulation_locations.csv")
    MERGE_KEYS_COORD = ['Latitude', 'Longitude']
    wq_train = combine_two_datasets(wq_train, flow_train, keys=MERGE_KEYS_COORD)
    
    # Validation/Test Data
    landsat_val_features = pd.read_csv('./data/original/landsat_features_validation.csv')
    terraclimate_val_features = pd.read_csv('./data/original/terraclimate_features_validation.csv')
    submission_template = pd.read_csv('./data/original/submission_template.csv')
    
    wq_test = combine_two_datasets(landsat_val_features, terraclimate_val_features, keys=MERGE_KEYS)
    
    flow_test = pd.read_csv("data/processed/submission_flow_accumulation_locations.csv")
    wq_test = combine_two_datasets(wq_test, flow_test, keys=MERGE_KEYS_COORD)
    
    return wq_train, wq_test, submission_template

def preprocess_and_predict(wq_train, wq_test):
    selected_features = [
        'swir22', 'NDMI', 'MNDWI', 'pet', 
        'Latitude', 'Longitude', 'Month', 
        'flow_accumulation', 'NDSI', 'NDWI', 'swir_ratio'
    ]
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Preprocess training
    wq_train['flow_accumulation'] = np.log1p(wq_train['flow_accumulation'])
    wq_train['Sample Date'] = pd.to_datetime(wq_train['Sample Date'], dayfirst=True)
    wq_train['Month'] = wq_train['Sample Date'].dt.month
    wq_train = calculate_indices(wq_train)
    
    # Preprocess test
    wq_test['flow_accumulation'] = np.log1p(wq_test['flow_accumulation'])
    wq_test['Sample Date'] = pd.to_datetime(wq_test['Sample Date'], dayfirst=True)
    wq_test['Month'] = wq_test['Sample Date'].dt.month
    wq_test = calculate_indices(wq_test)
    
    # NaN handling for test (ffill/bfill)
    wq_test = wq_test.sort_values(['Latitude', 'Longitude', 'Sample Date'])
    fill_cols = ['swir22', 'NDMI', 'MNDWI', 'NDSI', 'NDWI', 'swir_ratio']
    wq_test[fill_cols] = wq_test.groupby(['Latitude', 'Longitude'])[fill_cols].ffill().bfill()
    
    X_test = wq_test[selected_features]
    predictions = {}
    
    for target in targets:
        print(f"Predicting for {target}...")
        model_path = f'models/best_model_{target.replace(" ", "_")}.joblib'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            predictions[target] = y_pred # DIRECT PREDICTION
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
