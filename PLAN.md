# Project Plan: EY Challenge 2026 - Water Quality Prediction - COMPLETED (REVISED V2)

The previous submission (mean $R^2$ = -0.261) was likely caused by poor spatial generalization and unreliable features. This revised approach focused on spatial robustness.

**STATUS: ALL TASKS COMPLETED**

## 1. Feature Engineering & Selection
- [x] Used Benchmark features: `swir22`, `NDMI`, `MNDWI`, `pet`.
- [x] Added Spatial features: `Latitude`, `Longitude` (to capture geographic trends).
- [x] Added Temporal features: `Month` (to capture seasonality).
- [x] Removed log-transform on targets to reduce bias/instability.

## 2. Robust Validation Strategy
- [x] Switched to **GroupKFold** (grouped by Latitude/Longitude).
- [x] This ensured that validation scores reflect performance on unseen locations.
- [x] **GroupKFold R2 Results**:
    - **Total Alkalinity**: 0.3375
    - **Electrical Conductance**: 0.2971
    - **Dissolved Reactive Phosphorus**: 0.1170
- (Note: These are much more realistic than the previous 0.8+ scores obtained with leaked KFold).

## 3. Model Optimization
- [x] Implemented a robust **RandomForestRegressor** pipeline.
- [x] Hyperparameters tuned for generalization (max_depth=15, min_samples_leaf=5).

## 4. Test Set Inference
- [x] Updated `generate_submission.py` to use the 7-feature set.
- [x] Implemented neighborhood filling (`ffill`/`bfill`) for missing Landsat data.
- [x] Added clipping to prevent unrealistic/negative predictions.
- [x] Generated the final **`submission.csv`**.
