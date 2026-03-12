import os
import pandas as pd
import numpy as np
from pipeline import combine_two_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

def load_data():
    # Training Data
    Water_Quality_df = pd.read_csv('./data/original/water_quality_training_dataset.csv')
    landsat_train_features = pd.read_csv('./data/original/landsat_features_training.csv')
    Terraclimate_df = pd.read_csv('./data/original/terraclimate_features_training.csv')
    
    MERGE_KEYS = ['Latitude', 'Longitude', 'Sample Date']
    wq_train = combine_two_datasets(Water_Quality_df, landsat_train_features, Terraclimate_df, keys=MERGE_KEYS)
    
    # Validation/Test Data
    landsat_val_features = pd.read_csv('./data/original/landsat_features_validation.csv')
    terraclimate_val_features = pd.read_csv('./data/original/terraclimate_features_validation.csv')
    submission_template = pd.read_csv('./data/original/submission_template.csv')
    
    wq_test = combine_two_datasets(landsat_val_features, terraclimate_val_features, keys=MERGE_KEYS)
    
    return wq_train, wq_test, submission_template

def preprocess_and_train(wq_train, wq_test):
    # Benchmark features + Lat/Lon + Month
    selected_features = ['swir22', 'NDMI', 'MNDWI', 'pet', 'Latitude', 'Longitude', 'Month']
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Preprocess training
    wq_train['Sample Date'] = pd.to_datetime(wq_train['Sample Date'], dayfirst=True)
    wq_train['Month'] = wq_train['Sample Date'].dt.month
    
    # Preprocess test
    wq_test['Sample Date'] = pd.to_datetime(wq_test['Sample Date'], dayfirst=True)
    wq_test['Month'] = wq_test['Sample Date'].dt.month
    
    # Optional: neighborhood fill for spectral features
    wq_test = wq_test.sort_values(['Latitude', 'Longitude', 'Sample Date'])
    wq_test[['swir22', 'NDMI', 'MNDWI']] = wq_test.groupby(['Latitude', 'Longitude'])[['swir22', 'NDMI', 'MNDWI']].fillna(method='ffill').fillna(method='bfill')
    
    X_train = wq_train[selected_features]
    X_test = wq_test[selected_features]
    
    predictions = {}
    
    for target in targets:
        print(f"Loading/Training model for {target}...")
        y_train = wq_train[target]
        
        model_path = f'models/best_model_{target.replace(" ", "_")}.joblib'
        if os.path.exists(model_path):
            print(f"Loading saved model from {model_path}")
            model = joblib.load(model_path)
        else:
            print("No saved model found, using robust RandomForest")
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('rf', RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42))
            ])
            model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        predictions[target] = y_pred
        
    return predictions, wq_test

def main():
    wq_train, wq_test, submission_template = load_data()
    predictions, wq_test_processed = preprocess_and_train(wq_train, wq_test)
    
    results_df = wq_test_processed[['Latitude', 'Longitude', 'Sample Date']].copy()
    for target, preds in predictions.items():
        results_df[target] = preds
        
    # Ensure date format matches template for merging
    results_df['Sample Date'] = results_df['Sample Date'].dt.strftime('%d-%m-%Y')
    
    # Final merge to match template exactly
    final_sub = submission_template[['Latitude', 'Longitude', 'Sample Date']].merge(
        results_df, on=['Latitude', 'Longitude', 'Sample Date'], how='left'
    )
    
    # Ensure columns match template
    final_sub = final_sub[['Latitude', 'Longitude', 'Sample Date', 'Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']]
    
    # Fallback for any missing values
    for target in ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']:
        median_val = wq_train[target].median()
        final_sub[target] = final_sub[target].fillna(median_val)
        
        # Clip negative predictions
        min_val = wq_train[target].min()
        final_sub[target] = final_sub[target].clip(lower=min_val)
    
    final_sub.to_csv('submission.csv', index=False)
    print("Saved submission to submission.csv")

if __name__ == "__main__":
    main()
