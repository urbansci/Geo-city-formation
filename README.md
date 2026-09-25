# Urban Niche

This repository contains the implementation and data for the paper:
>**Self-supervised learning quantifies the geographical foundations of city formation**
>
>Weiyu Zhang, Lei Dong∗, Junjie Yang, and Yu Liu∗
>  

## Repository Structure

```
└──codes/                                   
      ├── data_processing/                       
      │   ├── 2nd_nature_geography/               # Second nature geography calculations
      │   ├── agriculture_data/                   # Agricultural potential dataset construction
      │   ├── climate_data_preprocessing/         # Climate data processing
      │   ├── manual_features/                    # Manual feature extraction
      │   └── pretraining_dataset_construction/   # Dataset construction for pretraining
      │      
      ├── mae/                                   # Masked Autoencoder folder
      │   ├── models/                            # Model architectures
      │   ├── util/                              # Utility functions
      │   ├── embedding_extraction.py            # Inference script
      │   └── pretrain.py                        # Pretraining script
      │   
      ├── plotting/                              # Visualization and figure generation
      │
      ├── statistical_analysis/                  
      │   ├── quantitative_analysis/             # Quantitative analysis tools
      │   │   ├── configs/                       # Configuration files
      │   │   ├── models/                        # XGBoost model trainer
      │   │   ├── utils/                         # Utility functions to prepare data for classifiers
      │   │   ├── future/                        # Scripts to conduct analysis in "Future projections of urban suitability"
      │   │   ├── temporal_analysis/             # Scripts to conduct analysis in "Historical dynamics of geographical influence on city formation"
      │   │   └── city_location_prediction/      # Scripts to conduct analysis in "Predicting city locations"
      │   └── qualitative_analysis/              # Scripts to conduct analysis in "Quantifying complex geographical environments via self-supervised learning" and "Low-dimensional structure of geographical embeddings"
      │ 
      ├── .gitignore                             # Git ignore file
      └── README.md                              # Project documentation
```

## Data

### Raw Data Sources

1. **Topography**: Forest and Buildings removed Copernicus Digital Elevation Model (FABDEM) v1-2
   - Resolution: 1 arc-second (~30m at equator)
   - Source: https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn

2. **Hydrology**: Global Surface Water (GSW) dataset
   - Resolution: 1 arc-second (~30m at equator)
   - Source: https://global-surface-water.appspot.com/

3. **Climate**: CHELSA (Climatologies at High Resolution for Earth's Land Surface Areas)
   - Historical: https://chelsa-climate.org/chelsa-trace21k/
   - Contemporary: https://chelsa-climate.org/
   - Future projections: https://chelsa-climate.org/cmip6/

4. **Agricultural Potential**: Global Agro-Ecological Zones (GAEZ) v4
   - Source: https://gaez.fao.org/

5. **Historical Cities**:
   - European cities (700-2000 CE): https://ssh.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-xzy-u62q
   - Chinese cities CHGIS (370 BCE-1911 CE): https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WW1PD6
   - Chinese walled cities (1368-1911 CE): https://doi.org/10.6084/m9.figshare.14112968.v3

### Processed Data
Our processed datasets include geographical embeddings from self-supervised learning models, manual geographical features, and spatially verified historical city data. See our [data repository](https://bit.ly/Data-geo-drivers-city-formation)(https://bit.ly/Data-geo-drivers-city-formation).

## Workflow
### Step 1: Image dataset construction
The first step is to build datasets for training or inference of self-supervised learning (SSL) models. Use the following commands to clip raw raster data into image datasets:

```bash
# Process current climate data
# For other data types, refer to clip_images_with_grids_multiband.py parameter list
python -u codes/data_processing/pretraining_dataset_construction/clip_grids/clip_images_with_grids_multiband.py current_clim \
    --ssp "" \
    --year 2010 \
    --grid_path "/path/to/your/grid.shp" \
    --output "/path/to/datasets/climate/current_clim_grid"
```
### Step 2: Geographical Embedding Extraction
Use the following commands to launch pretraining. 

#### Terrain-Water Model Pretraining
```bash
python codes/mae/pretrain.py \
    --batch_size 600 \
    --model mae_vit_large_patch16 \
    --mask_ratio 0.75 \
    --num_workers 8 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path "/path/to/datasets/terrain_water" \
    --output_dir "/path/to/saved_models/terrain_water_model" \
    --in_chans 2 \
    --img_size 160 \
    --patch_size 16 \
    --embed_dim 256 \
    --dist_url "tcp://localhost:10001" \
    --multiprocessing_distributed \
    --world_size 1 \
```
#### Climate Model Pretraining
```bash
python codes/mae/pretrain.py \
    --batch_size 600 \
    --model mae_vit_large_patch16 \
    --mask_ratio 0.75 \
    --num_workers 8 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path "/path/to/datasets/climate" \
    --output_dir "/path/to/saved_models/climate_model" \
    --in_chans 9 \
    --img_size 24 \
    --patch_size 4 \
    --embed_dim 128 \
    --dist_url "tcp://localhost:10001" \
    --multiprocessing_distributed \
    --world_size 1 \
```
### Step3: Embedding extaction
Extract geographical embeddings using pre-trained self-supervised models:
```bash
# Terrain-Water Embeddings ; See args in codes/mae/embedding_extraction.py for other types of embeddings extraction.
python codes/mae/embedding_extraction.py \
    --gpu 0 \
    --batch_size 1600 \
    --workers 8 \
    --save_root "/path/to/embeddings/terrain_water" \
    --seed 42 \
    --pretrained "/path/to/saved_models/terrain_water_model/checkpoint-499.pth" \
    --base_name "terrain_water_embeddings_256d" \
    --model "vit_large_patch16" \
    --file_list "/path/to/file_lists/terrain_water_locations.parquet" \
    --dem 
```
### Step4: Statistical analysis
#### Section 2: Low-dimensional structure of geographical embeddings
run `codes/statistical_analysis/qulitative_analysis/embedding_space_analysis.ipynb` to conduct dimensionality reduction using UMAP and KMeans Clustering to embeddings. 
#### Section 3: Low-dimensional structure of geographical embeddings
run `codes/statistical_analysis/qulitative_analysis/embedding_space_analysis.ipynb` to conduct dimensionality reduction using UMAP and KMeans Clustering to embeddings. 

#### Section 4: Predicting city locations
run following command to launch XGboost training and evaluation:
```bash
python codes/statistical_analysis/quantitative_analysis/city_location_prediction/base_analysis_bootstrapping.py \
    -d cn-walled \ #['cn-pref','eu','cn-wlled']
    --feature_type All_embedding \ #['All_embedding','DEM_water_embedding','Climate_embedding','Manual_feature']
    --bootstrap_iterations 1000 \
    --enable_shap \
    --output_path ./results

```
run `codes/plotting/plot_model_preformance.ipynb` to plot performance barchart (Figure 2). 

#### Section 5: Historical dynamics of geographical influence on city formation
- run `codes/statistical_analysis/quantitative_analysis/temporal_analysis/temporal_analysis.py` to launch temporal prediction tasks via sliding window.

#### Section 6:Future projections of urban suitability
- run `codes/statistical_analysis/quantitative_analysis/future_projection/base_analysis_combining_eu_cn_bootstrapping_n300.py` to train XGboost model.
- run `codes/statistical_analysis/quantitative_analysis/future_projection/XGBoost_inference_ensemble_global_current.py`and `codes/statistical_analysis/quantitative_analysis/future_projection/XGBoost_inference_ensemble_global_future.py` to conduct inference using trained XGBoost model across the globe.
- run `codes/statistical_analysis/quantitative_analysis/future_projection/comparison_future_current.py` to obtain the urban potential change. 
- run `codes/plotting/future_projection/prepare_prob_change_bins_by_latlon.ipynb` and `codes/plotting/future_projection/plot_prob_change_latlon.py` to plot ubran potential changes along the latitude and longitude. 

## Citation
If you use this code or data in your research, please cite:

```bibtex
TBD
```

## Contact
If you have any questions, feel free to contact us through email wyzhang929@gmail.com.
