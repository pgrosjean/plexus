# Plexus
**Overview**: Network-Aware Masked Autoencoders for Neuronal Activity Phenotyping
<img width="813" alt="image" src="https://github.com/user-attachments/assets/7558784b-bb7e-4fc8-b331-2d6a8f3b5d4c" />


# Installation
```bash
git clone https://github.com/pgrosjean/plexus.git
cd plexus
bash setup_mamba.sh
conda activate plexus
pip install -e .
```

# Hardware and Software Specifications
All training and inference requires at least one NVIDIA GPU with at least 24 Gb memory

# Downloading the data archive
To train the models you must first download the data archive from Zenodo
Zenodo DOI (10.5281/zenodo.14714574)
```bash
wget https://zenodo.org/records/14714574
gunzip plexus_data_archive.gz
```
**Note: this data archive also includes .h5ad files containing both the manual features and the plexus embeddings.**

# Training models
Before training any models ensure that you set up wandb
```bash
wandb login
```
### CRISRPi Screen Plexus Model Training
Change the logging directory if desired, default below will save files to ./logging/
```bash
plexus-train --config crispri_screen_8cell --log_dir ./logging/
```
### CRISRPi Screen 1 cell MAE Model Training
```bash
plexus-train --config crispri_screen_1cell --log_dir ./logging/
```
### Neuroactive Stimulation Model Training
```bash
plexus-train --config neuroactive_8cell --log_dir ./logging/
```
### Simulation Model Training
```bash
plexus-train --config simulation_8cell --log_dir ./logging/
```

# Running inference for models
When running inference this assumes you have properly downloaded the dataset archive folder and are in the base path of the plexus repository.

If you are running from another location you will need to change the paths to match your working directory.

Running inference generates parquet files with the embeddings infromation and plate and well information.


### CRISRPi Screen Plexus Model Inference
```bash
plexus-inference --config crispri_screen_8cell --zarr_path ./plexus_data_archive/processed_zarr_files/crispri_screen/split_zarr_files/ --dataset_stats_json ./plexus_data_archive/dataset_statistics/crispri_screen/CRISPRI_SCREEN_DATASET_STATS_DICT.json --only_nuclei_positive --save_path ./crispri_screen_embedding_parquet_files/ --checkpoint_path ./plexus_data_archive/model_checkpoints/crispri_screen/crispri_screen_8cell/model-72o1c2vc:v1/model.ckpt
```

### CRISRPi Screen 1 Cell MAE Model Inference
```bash
plexus-inference --config crispri_screen_1cell --zarr_path ./plexus_data_archive/processed_zarr_files/crispri_screen/split_zarr_files/ --dataset_stats_json ./plexus_data_archive/dataset_statistics/crispri_screen/CRISPRI_SCREEN_DATASET_STATS_DICT.json --only_nuclei_positive --save_path ./crispri_screen_embedding_parquet_files/ --checkpoint_path ./plexus_data_archive/model_checkpoints/crispri_screen/crispri_screen_1cell/model-gmzj27s2:v1/model.ckpt
```

### Simulation Plexus Model Inference
```bash
plexus-inference --config simulation_8cell --zarr_path ./plexus_data_archive/processed_zarr_files/simulation/ --dataset_stats_json ./plexus_data_archive/dataset_statistics/simulation/SIMULATION_STATS_DICT.json --save_path ./simulation_embedding_parquet_files/ --checkpoint_path ./plexus_data_archive/model_checkpoints/simulation/model-1s3n8lon:v1/model.ckpt
```
