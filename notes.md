## Land use/ land cover approach
I plan to identify the percentage of Urban, Agricultural, Mining and Industrial use

## Other remote sensing index: https://support.climateengine.org/article/103-metrics
I should use all of them to predict





- https://zenodo.org/records/14195737: Mine segmentation level 
- https://www.icmm.com/en-gb/research/data/2025/global-mining-dataset: Global mining dataset
- https://allenai.org/blog/satlaspretrain-models-foundation-models-for-satellite-and-aerial-imagery-1679ebe4bbfb: Pre trained model identifying satelite image
- https://www.mdpi.com/2072-4292/14/22/5657: FE4395 dataset, industrial 
- https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200?_gl=1*cygdoc*_up*MQ..*_ga*MjM2NTU5MzcwLjE3Njk5Mjk2MTI.*_ga_SM8HXJ53K2*czE3Njk5Mjk2MTEkbzEkZzAkdDE3Njk5Mjk2MTEkajYwJGwwJGgw#terms-of-use: ESA worldcover v200


## Current problem 
- [] Massive overfit: R2 train = 0.9; R2 validation = 0.5
- Solution (implemented in pipeline + notebook): 
    - PCA and feature selection with mutual information
    - **Regularization**: `DEFAULT_REGULARIZED_XGB` (max_depth=4, reg_alpha/lambda, subsample)
    - **use_pca=False** for tree-only pipeline (often better with few features)
    - **Finetuning**: `run_pipeline_with_tuning()` (GridSearchCV over max_depth, lr, reg)
    - Use **all 7 features** (nir, green, swir16, swir22, NDMI, MNDWI, pet) in preprocessing
    - Gather more training data
    
