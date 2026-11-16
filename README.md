# EY Challenge 2026: water quality from satellite data

Predicts three water-quality measures at South African sampling sites from Landsat bands, TerraClimate, elevation and flow accumulation: total alkalinity, electrical conductance and dissolved reactive phosphorus.

<!-- demo: add docs/demo.gif and an image link here once the recording exists -->

## Architecture

```
water_quality_training_dataset.csv  ─┐
Landsat bands (swir22, NDMI, MNDWI)  ─┼─▶ pipeline.combine_two_datasets ─▶ spectral indices ─▶ imputer ─▶ scaler ─▶ RandomForest ─▶ submission.csv
TerraClimate (pet)                   ─┤                                    (NDSI, NDWI,                                (one per
flow accumulation (Earth Engine)     ─┘                                     swir_ratio) + month,                          target)
                                                                            lat/lon bins
```

| File | Role |
|---|---|
| `pipeline.py` | Dataset merging, an XGBoost pipeline with RFE and PCA, out-of-fold training with Weights & Biases learning curves, Earth Engine flow-accumulation sampling |
| `optimize_model.py` | Final model: random forest per target, tuned with `RandomizedSearchCV` under location-grouped 5-fold CV |
| `generate_submission.py` | Builds test features and writes `submission.csv` (200 rows) from the saved models |
| `visualize_project.py` | Feature-importance plot and geospatial map |
| `notebook/` | Earth Engine and Landsat extraction notebooks, plus the challenge benchmark notebook |
| `run_main.slurm` | Job script for the Compute Canada cluster |

The training set has 9,319 samples from 162 locations (2013 to 2015). The test set has 200 rows from 24 locations, and none of them appears in training. The final model therefore validates with `GroupKFold` grouped by location. An earlier XGBoost pipeline overfit: train R² about 0.9 against validation about 0.5 (`notes.md`).

## Results

Best score from `optimize_model.py` (random forest, 13 features, `GroupKFold` with 5 splits, R²), reproduced on 2026-11-16:

| Target | R² |
|---|---|
| Total alkalinity | 0.25 |
| Electrical conductance | 0.36 |
| Dissolved reactive phosphorus | 0.10 |

These are cross-validation scores on training locations. The repo has no leaderboard score for `submission.csv`. Phosphorus is the weakest target.

## Quickstart

```bash
pip install numpy pandas scikit-learn joblib xgboost matplotlib wandb
export WANDB_MODE=disabled
python optimize_model.py        # trains and saves models/best_model_<target>.joblib
python generate_submission.py   # writes submission.csv
```

`requirements.txt` is the full pinned environment from the cluster. `visualize_project.py` also needs `seaborn` and `geopandas`. Earth Engine sampling in `pipeline.py` needs an authenticated `earthengine-api` session.
