# Project Plan: EY Challenge 2026 - Water Quality Prediction - COMPLETED

This plan outlines the steps to optimize the water quality prediction models, generate submissions for the private test set, and create a visual map for project showcase.

## 1. Research & Baseline Verification
- [x] Fix minor bugs in `pipeline.py` (e.g., missing imports like `StandardScaler`).
- [x] Reproduce current baseline results using `main.ipynb` logic.
- [x] Identify bottleneck features and target variables with low performance (e.g., Dissolved Reactive Phosphorus).

## 2. Model Optimization (Auto-Research Inspired)
- [x] **Feature Engineering Automation**:
    - [x] Evaluated additional remote sensing bands (nir, green, swir16).
    - [x] Identified that simple feature set is more robust for current dataset size.
- [x] **Hyperparameter Tuning**:
    - [x] Implemented `optimize_model.py` using `RandomizedSearchCV` for XGBoost and RandomForest.
    - [x] Found optimized hyperparameters for each target parameter.
- [x] **Ensembling**:
    - [x] Built a VotingRegressor ensemble of optimized RandomForest and XGBoost models.
    - [x] Achieved improved OOF $R^2$ scores (e.g., 0.865 for Alkalinity).

## 3. Inference on Test Set
- [x] Prepare the test features for `data/original/submission_template.csv`.
    - [x] Merged validation features from Landsat and TerraClimate.
    - [x] Handled missing land use data via median imputation from training set.
- [x] Run the optimized ensemble model on the test set.
- [x] Format the output according to the `submission_template.csv` and saved to `submission.csv`.

## 4. Visualization & Showcase
- [x] **Geospatial Map**:
    - [x] Created `visualize_project.py` using `geopandas` to show sampling locations colored by target concentration.
    - [x] Saved results to `visuals/geospatial_map.png`.
- [x] **Model Performance Dashboard**:
    - [x] Generated feature importance plots for the ensemble models.
    - [x] Saved results to `visuals/feature_importance.png`.

## 5. Final Documentation
- [x] Updated `PLAN.md` with completion status.
- [x] Provided scripts (`optimize_model.py`, `generate_submission.py`, `visualize_project.py`) for reproducibility.
