import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import joblib

def load_data():
    Water_Quality_df = pd.read_csv('./data/original/water_quality_training_dataset.csv')
    return Water_Quality_df

def plot_geospatial(df):
    print("Generating geospatial map...")
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    
    # Set CRS to WGS84
    gdf.set_crs(epsg=4326, inplace=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    for i, target in enumerate(targets):
        ax = axes[i]
        
        # Plot sampling points
        gdf.plot(column=target, ax=ax, cmap='viridis', legend=True, 
                 legend_kwds={'label': target, 'orientation': "horizontal"},
                 markersize=30, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_title(f'Sampling Locations: {target}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig('visuals/geospatial_map.png')
    print("Saved visuals/geospatial_map.png")

def plot_feature_importance():
    print("Generating feature importance plots...")
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    features = ['swir22', 'NDMI', 'MNDWI', 'pet', "flow_accumulation", "pct_agricultural", "pct_urban"]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, target in enumerate(targets):
        model_path = f'models/best_model_{target.replace(" ", "_")}.joblib'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            # Assuming it's a Pipeline with an 'ensemble' step (VotingRegressor)
            # We'll take the average importance of RF and XGB if possible,
            # but VotingRegressor doesn't have feature_importances_.
            # Let's try to get it from the individual models.
            ensemble = model.named_steps['ensemble']
            rf_model = ensemble.estimators_[0]
            xgb_model = ensemble.estimators_[1]
            
            importances = (rf_model.feature_importances_ + xgb_model.feature_importances_) / 2
            
            sns.barplot(x=importances, y=features, ax=axes[i], palette='magma')
            axes[i].set_title(f'Feature Importance: {target}')
        else:
            axes[i].text(0.5, 0.5, 'Model not found', ha='center')
            
    plt.tight_layout()
    plt.savefig('visuals/feature_importance.png')
    print("Saved visuals/feature_importance.png")

def main():
    os.makedirs('visuals', exist_ok=True)
    df = load_data()
    plot_geospatial(df)
    plot_feature_importance()

if __name__ == "__main__":
    main()
