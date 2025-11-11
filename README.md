# AIGS Models to PyTorch

This repository contains PyTorch reimplementations of two published deep learning models for genomic selection. DNNGP was developed by Wang et al. (2023) and written in TensorFlow 2.6.0. DeepGS was developed by Ma et al. (2018) and written in MXNet. Default parameters may differ slightly from publications to based on reoptimization using the code provided here.

## Overview

This project implements convolutional neural network architectures for predicting phenotypic traits from genomic data. The models use SNP (Single Nucleotide Polymorphism) data to predict quantitative traits such as grain length.

## Models

The implementations here of the two models are:

### DNNGP (Deep Neural Network Genomic Prediction)
- Architecture: 1D Convolutional Neural Network with 3 convolutional layers
- Input: Principal Components (PCs) of SNP data or raw SNP data
- Features: Batch normalization, dropout regularization

### DeepGS
- Architecture: 1D Convolutional Neural Network with max pooling
- Input: Principal Components (PCs) of SNP data or raw SNP data
- Features: Max pooling, dropout regularization

## Project Structure

```
.
├── Data/                  # Input data directory (user-provided)
├── Output/                # Output directory (generated)
├── DNNGP_data.py          # Data preprocessing and preparation
├── DNNGP_model.py         # DNNGP model architecture
├── DeepGS_model.py        # DeepGS model architecture
├── Train.py               # Training functions with k-fold cross-validation
├── Test.py                # Extended testing with early stopping
├── Kfold_train.py         # K-fold cross-validation training script
├── Simple_train.py        # Simple training script with validation split
├── ObsVsPred_plot.py      # Visualization of observed vs predicted values
├── ObsVsPred100.py        # Analysis script for multiple runs
├── RunData.sh             # Shell script for data preprocessing
├── RunModel.sh            # Shell script for model training
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AIGSModels2Pytorch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

First, prepare your data using `DNNGP_data.py` + custom parameters or the shell script:

```bash
# Using Python Directly:
python DNNGP_data.py \
    --data_dir "./Data" \
    --output_dir "./Output" \
    --snp_start_col 20000 \
    --snp_end_col 26280 \
    --min_allele_freq 0.05 \
    --max_allele_freq 0.95 \
    --max_geno_freq 0.95

# or edit the bash script with desired parameters
bash RunData.sh 
```

This script:
- Loads phenotypic and genotypic data
- Performs quality control (removes duplicates, filters SNPs)
- Applies allele frequency filtering
- Saves processed data to `./Output/DNNGP_input.pt`

**Available flags for DNNGP_data.py:**
- `--data_dir`: Directory containing input data files (default: ./Data)
- `--output_dir`: Directory for output files (default: ./Output)
- `--snp_start_col`: Starting column index for SNP data subset (default: 20000)
- `--snp_end_col`: Ending column index for SNP data subset (default: 26280)
- `--min_allele_freq`: Minimum allele frequency threshold (default: 0.05)
- `--max_allele_freq`: Maximum allele frequency threshold (default: 0.95)
- `--max_geno_freq`: Maximum genotype frequency threshold (default: 0.95)


### Training

#### K-fold Cross-Validation Training
```bash
# Using Python directly
python Kfold_train.py \
    --folds 3 \
    --epochs 5 \
    --batch_size 32 \
    --PCs 1000 \
    --model_type "DeepGS" \
    --lr 0.00001 \
    --momentum 0.5 \
    --weight_decay 0.00001 \
    --input_file "./Output/DNNGP_input.pt"

# Or using the shell script with desired parameters
bash RunModel.sh
```

**Available flags for Kfold_train.py:**
- `--folds`: Number of k-fold splits (default: 10)
- `--epochs`: Number of training epochs (default: 400)
- `--batch_size`: Batch size (default: 17)
- `--PCs`: Number of principal components (default: 1691)
- `--model_type`: Model type: DNNGP or DeepGS (default: DNNGP)
- `--lr`: Learning rate (default: 0.00001)
- `--momentum`: SGD momentum (default: 0.5)
- `--weight_decay`: Weight decay (default: 0.00001)
- `--input_file`: Input data file path (default: ./Output/DNNGP_input.pt)

#### Extended Training with Early Stopping
```bash
python Test.py
```

#### Simple Training with Validation Split
```bash
# Using Python directly with default parameters
python Simple_train.py

# Or with custom parameters
python Simple_train.py --epochs 5 --batch_size 32 --model_type DNNGP --lr 0.001 --validation_split 0.15
```

**Available flags for Simple_train.py:**
- `--input_file`: Input data file path (default: ./Output/DNNGP_input.pt)
- `--validation_split`: Validation split ratio (default: 0.10)
- `--model_type`: Model type: DNNGP or DeepGS (default: DNNGP)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 20)
- `--lr`: Learning rate (default: 0.01)
- `--momentum`: SGD momentum (default: 0.5)
- `--weight_decay`: Weight decay (default: 0.00001)
- `--PCs`: Number of principal components (default: None, uses input_data.shape[2] - 1)

### Visualization

Generate plots of observed vs predicted values:
```bash
# Using Python directly with desired parameters
python ObsVsPred_plot.py \
    --results_file "./Output/Validation_output.npy" \
    --metadata_file "./Output/metadata.npy" \
    --output_dir "./Output" \
    --title "Grain Length" \
    --folds 3
```

**Available flags for ObsVsPred_plot.py:**
- `--results_file`: Path to validation output results file (default: ./Output/Validation_output.npy)
- `--metadata_file`: Path to metadata file (default: ./Output/metadata.npy)
- `--output_dir`: Directory for output plots (default: ./Output)
- `--title`: Title for the plots (default: Grain Length)
- `--folds`: Number of folds (default: None, auto-detect from results)

## Configuration

Key parameters that can be adjusted in the training scripts:

- `folds`: Number of k-fold splits (default: 10)
- `epochs`: Maximum number of training epochs (default: 400-1000)
- `batch_size`: Batch size for training (default: 17-25)
- `PCs`: Number of principal components (default: 1691)
- `lr`: Learning rate (default: 0.00001-0.0001)
- `momentum`: SGD momentum (default: 0.5)
- `weight_decay`: L2 regularization (default: 0.00001)

## Output

The training scripts generate:
- Model checkpoints: `./Output/Model-fold/model-fold-{fold}.pth`
- Validation results: `./Output/Validation_output.npy`
- Loss curves: `./Output/Loss_curve.npy`
- Metadata: `./Output/metadata.npy`
- Visualization plots: `./Output/obs_pred_*.png`, `./Output/Loss.png`

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for full list of dependencies

## Data Format

The input data should be organized as follows:
- Phenotypic data: Excel files with columns for genotype IDs and trait values
- Genotypic data: CSV files with SNP data (rows = SNPs, columns = genotypes)
- Data should be placed in `./Data/` directory


## Citations

Original papers:

- DeepGS: Ma, W., Qiu, Z., Song, J., Li, J., Cheng, Q., Zhai, J. and Ma, C. A deep convolutional neural network approach for predicting phenotypes from genotypes. Planta 248, 1307–1318 (2018). https://doi.org/10.1007/s00425-018-2976-9

- DNNGP: Wang, K., Abid, M.A., Rasheed, A., Crossa, J., Hearne, S. and Li, H. DNNGP, a deep neural network-based method for genomic prediction using multi-omics data in plants. Molecular Plant 16, 279-293 (2023). https://doi.org/10.1016/j.molp.2022.11.004