# %% [markdown]
# ## Predictor Variable
# <p align="justify">
# Now that we have our water quality dataset, the next step is to gather the predictor variables from the <b>Landsat</b> and <b>TerraClimate</b> datasets. In this notebook, we demonstrate how to <b>load previously extracted satellite and climate data</b> from separate files, rather than performing the extraction directly, which allows for a smoother and faster experience. Participants can refer to the dedicated extraction notebooks—one for Landsat and another for TerraClimate—to understand how the data was retrieved and processed, and they can also generate their own output CSV files if needed. Using these pre-extracted CSV files, this notebook focuses on loading the predictor features and running the subsequent analysis and model training efficiently.
# </p>
# <p align="justify">
# For more detailed guidance on the original data extraction process, you can review the <a href="https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2#Example-Notebook">Landsat example notebook</a> and the <a href="https://planetarycomputer.microsoft.com/dataset/terraclimate#Example-Notebook">TerraClimate example notebook</a> available on the Planetary Computer portal.
# </p>
# 
# <p align="justify">We have used selected spectral bands — SWIR22 (Shortwave Infrared 2), NIR (Near Infrared), Green, and SWIR16 (Shortwave Infrared 1) — and computed key spectral indices such as NDMI (Normalized Difference Moisture Index) and MNDWI (Modified Normalized Difference Water Index). These features capture surface moisture, vegetation, and water content characteristics that influence water quality variability. </p> <p align="justify"> In addition to Landsat features, we also incorporated the <b>Potential Evapotranspiration (PET)</b> variable from the <b>TerraClimate</b> dataset, which provides high-resolution global climate data. The PET feature captures the atmospheric demand for moisture, representing climatic conditions such as temperature, humidity, and radiation that influence surface water evaporation and thus affect water quality parameters. </p> <ul> <li>SWIR22 – Sensitive to surface moisture and turbidity variations in water bodies.</li> <li>NIR – Helps in identifying vegetation and suspended matter in water.</li> <li>Green – Useful for detecting water color and surface reflectance changes.</li> <li>SWIR16 – Provides information on surface dryness and sediment concentration.</li> <li>NDMI – Derived from NIR and SWIR16, indicates moisture and vegetation-water interaction.</li> <li>MNDWI – Derived from Green and SWIR22, effective for distinguishing open water areas and reducing built-up noise.</li> <li>PET – Extracted from the TerraClimate dataset, represents the potential evapotranspiration that influences hydrological and water quality dynamics.</li> </ul>

import pandas as pd
import wandb

# %%
%load_ext autoreload
%autoreload 2

# %% [markdown]
# ## Load data

# %%
Water_Quality_df = pd.read_csv('./data/original/water_quality_training_dataset.csv')
Water_Quality_df.head()

landsat_train_features = pd.read_csv('./data/original/landsat_features_training.csv')
landsat_train_features.head()

# %%
Terraclimate_df = pd.read_csv('./data/original/terraclimate_features_training.csv')
Terraclimate_df.head()

# %%
from utils.pipeline import *
MERGE_KEYS = ['Latitude', 'Longitude', 'Sample Date']
wq_data = combine_two_datasets(Water_Quality_df, landsat_train_features, Terraclimate_df, keys=MERGE_KEYS)
wq_data.head()

# %% [markdown]
# ## Preprocess data

 # %%
wq_data = wq_data[['swir22','NDMI','MNDWI','pet', 'Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']]
wq_data

# %% [markdown]
# ## Run pipeline
# 

# %%
X = wq_data.drop(columns=['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus'])

y_TA = wq_data['Total Alkalinity']
y_EC = wq_data['Electrical Conductance']
y_DRP = wq_data['Dissolved Reactive Phosphorus']

# Initialize a Weights & Biases run for this script
wandb.init(
    project="ey_challenge_2026",
    name="main_full_and_oof",
    config={
        "features": list(X.columns),
        "pipelines": ["full_single_split", "simple_oof"],
    },
)

model_TA, results_TA = run_pipeline(X, y_TA, "Total Alkalinity", pipeline_kind='full')
model_EC, results_EC = run_pipeline(X, y_EC, "Electrical Conductance", pipeline_kind='full')
model_DRP, results_DRP = run_pipeline(X, y_DRP, "Dissolved Reactive Phosphorus", pipeline_kind='full')

# %% [markdown]
# ## Out-of-fold validation (OOF)
# 
# Like the reference code: train one model per fold, fill OOF predictions for the training set, and (optionally) average fold models for test predictions. This gives a more reliable estimate of generalization and can improve scores.
# 
# - **`pipeline_kind='simple'`** – benchmark-style: imputer + scaler + RandomForest (no PCA). Often yields higher test R² than the full PCA+XGBoost pipeline.
# - **`pipeline_kind='full'`** – current pipeline: imputer + scaler + PCA + XGBoost.
# - Pass **`X_test`** to get averaged test predictions from all fold models.

# %%
from utils.pipeline import run_pipeline_oof

N_SPLITS = 5
SEED = 42

# OOF with simple (benchmark-style) pipeline – often better test R² than full pipeline
models_TA, oof_TA, results_TA_oof, _ = run_pipeline_oof(X, y_TA, n_splits=N_SPLITS, param_name="Total Alkalinity", pipeline_kind='simple', random_state=SEED)
models_EC, oof_EC, results_EC_oof, _ = run_pipeline_oof(X, y_EC, n_splits=N_SPLITS, param_name="Electrical Conductance", pipeline_kind='simple', random_state=SEED)
models_DRP, oof_DRP, results_DRP_oof, _ = run_pipeline_oof(X, y_DRP, n_splits=N_SPLITS, param_name="Dissolved Reactive Phosphorus", pipeline_kind='simple', random_state=SEED)

# Optional: with held-out X_test, pass it to get averaged test predictions
# models_TA, oof_TA, results_TA_oof, pred_test_TA = run_pipeline_oof(X, y_TA, X_test=X_test, n_splits=N_SPLITS, ...)


