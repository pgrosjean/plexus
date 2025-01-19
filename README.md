# Plexus
**Overview**: Network-Aware Masked Autoencoders for Neuronal Activity Phenotyping

# Installation
```bash
git clone https://github.com/pgrosjean/plexus.git
cd plexus
bash setup_mamba.sh
conda activate plexus
pip install -e .
```

# Downloading the datasets for training the models
To train the models you must first download the Datasets (zarr files) from Zenodo
```bash
wget https://zenodo.org/record/1234567/files/plexus_data_archive.gz
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
When running inference this assumes you have properly downloaded the dataset archive folder and are in the base path of the plexus repository
If you are running from another location you will need to change the paths to match your working directory.

### CRISRPi Screen Plexus Model Inference
```bash
plexus-inference --config crispri_screen_8cell --zarr_path ./plexus_data_archive/processed_zarr_files/crispri_screen/split_zarr_files/ --dataset_stats_json ./plexus_data_archive/dataset_statistics/crispri_screen/CRISPRI_SCREEN_DATASET_STATS_DICT.json --only_nuclei_positive
```
### Neuroactive Stimulation Plexus Model Inference
```bash
plexus-inference --config neuroactive_8cell --zarr_path ./plexus_data_archive/processed_zarr_files/neuroactive_stimulation/split_zarr_files/ --dataset_stats_json ./plexus_data_archive/dataset_statistics/neuroactive_stimulation/NEUROACTIVE_DATASET_STATS_DICT.json
```
### Simulation Plexus Model Inference
```bash
plexus-inference --config simulation_8cell --zarr_path ./plexus_data_archive/processed_zarr_files/siimulation/ --dataset_stats_json ./plexus_data_archive/dataset_statistics/simulation/SIMULATION_STATS_DICT.json
```
