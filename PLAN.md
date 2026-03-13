# Project Plan: EY Challenge 2026 - Water Quality Prediction - COMPLETED (REVISED V4)

We successfully maintained the spatial robustness of the 0.339 baseline while adding new requested spectral indices.

**STATUS: ALL TASKS COMPLETED**

## 1. Feature Engineering (New Spectral Indices)
- [x] Implemented **NDSI** (Salinity/Snow proxy): $(Green - SWIR16) / (Green + SWIR16)$.
- [x] Implemented **NDWI** (Water proxy): $(Green - NIR) / (Green + NIR)$.
- [x] Added **swir_ratio**: $SWIR22 / SWIR16$.
- [x] Maintained core features: `swir22`, `NDMI`, `MNDWI`, `pet`, `Latitude`, `Longitude`, `Month`, `flow_accumulation`.

## 2. Robust Model Strategy
- [x] Used 100% **RandomForestRegressor** for spatial stability.
- [x] **NO Target Log-Transform**: Predicted targets directly to avoid exponential back-transformation errors.
- [x] **GroupKFold Validation**: Grouped by location to ensure generalization to unseen regions.
- [x] **GroupKFold R2 Results**:
    - **Total Alkalinity**: 0.3096
    - **Electrical Conductance**: 0.2891
    - **Dissolved Reactive Phosphorus**: 0.0763

## 3. Test Set Inference
- [x] Updated `generate_submission.py` to calculate new indices for the test set.
- [x] Maintained neighborhood filling (`ffill`/`bfill`) for missing Landsat bands.
- [x] Added clipping to training set min/max to prevent outliers.
- [x] Generated the final **`submission.csv`**.

## 4. Visualization
- [x] Updated `visualize_project.py` to reflect the 11-feature importance dashboard.
- [x] Geospatial maps updated.
