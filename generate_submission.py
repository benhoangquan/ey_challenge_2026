import os
import pandas as pd
import numpy as np
from pipeline import combine_two_datasets
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

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
    
    land_train = pd.read_csv("data/processed/land_use_worldcover_1km_locations.csv")
    wq_train = combine_two_datasets(wq_train, land_train, keys=MERGE_KEYS_COORD)
    
    # Validation/Test Data
    landsat_val_features = pd.read_csv('./data/original/landsat_features_validation.csv')
    terraclimate_val_features = pd.read_csv('./data/original/terraclimate_features_validation.csv')
    submission_template = pd.read_csv('./data/original/submission_template.csv')
    
    wq_test = combine_two_datasets(landsat_val_features, terraclimate_val_features, keys=MERGE_KEYS)
    
    flow_test = pd.read_csv("data/processed/submission_flow_accumulation_locations.csv")
    wq_test = combine_two_datasets(wq_test, flow_test, keys=MERGE_KEYS_COORD)
    
    # Add missing land use columns to wq_test
    # We will impute these later using training medians
    wq_test['pct_urban'] = np.nan
    wq_test['pct_agricultural'] = np.nan
    
    return wq_train, wq_test, submission_template

def preprocess_and_train(wq_train, wq_test):
    features = ['swir22', 'NDMI', 'MNDWI', 'pet', "flow_accumulation", "pct_agricultural", "pct_urban"]
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Preprocess training
    wq_train['flow_accumulation'] = np.log1p(wq_train['flow_accumulation'])
    
    pt = PowerTransformer(method='box-cox', standardize=True)
    wq_train['pct_agricultural'] = pt.fit_transform(wq_train[['pct_agricultural']] + 1)
    wq_train['pct_urban'] = pt.fit_transform(wq_train[['pct_urban']] + 1)
    
    # Preprocess test
    wq_test['flow_accumulation'] = np.log1p(wq_test['flow_accumulation'])
    
    # For land use in test, we impute with training median *before* PowerTransform
    # Actually, we should impute with the transformed training median
    agri_median = wq_train['pct_agricultural'].median()
    urban_median = wq_train['pct_urban'].median()
    
    wq_test['pct_agricultural'] = agri_median
    wq_test['pct_urban'] = urban_median
    
    X_train = wq_train[features]
    X_test = wq_test[features]
    
    predictions = {}
    
    for target in targets:
        print(f"Training ensemble for {target}...")
        y_train = np.log1p(wq_train[target])
        
        # Best parameters found from optimization (manual approximation or use joblib)
        # For simplicity and robustness, I'll use the ensemble with reasonable defaults
        # or load the ones I just saved.
        
        model_path = f'models/best_model_{target.replace(" ", "_")}.joblib'
        if os.path.exists(model_path):
            print(f"Loading saved model from {model_path}")
            model = joblib.load(model_path)
        else:
            print("No saved model found, using default ensemble")
            rf = RandomForestRegressor(n_estimators=300, random_state=42)
            xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('ensemble', VotingRegressor([('rf', rf), ('xgb', xgb_model)]))
            ])
            model.fit(X_train, y_train)
            
        y_pred_log = model.predict(X_test)
        predictions[target] = np.expm1(y_pred_log)
        
    return predictions

def main():
    wq_train, wq_test, submission_template = load_data()
    predictions = preprocess_and_train(wq_train, wq_test)
    
    # Fill submission template
    # The order of rows in wq_test should match submission_template because we merged them carefully
    # But let's be sure by merging on keys
    
    results_df = wq_test[['Latitude', 'Longitude', 'Sample Date']].copy()
    for target, preds in predictions.items():
        results_df[target] = preds
        
    # Final merge to match template exactly
    final_sub = submission_template[['Latitude', 'Longitude', 'Sample Date']].merge(
        results_df, on=['Latitude', 'Longitude', 'Sample Date'], how='left'
    )
    
    # Ensure columns match template
    final_sub = final_sub[['Latitude', 'Longitude', 'Sample Date', 'Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']]
    
    final_sub.to_csv('submission.csv', index=False)
    print("Saved submission to submission.csv")

if __name__ == "__main__":
    main()
