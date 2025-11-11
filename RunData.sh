#!/bin/bash

# Run the data preprocessing script with arguments
python DNNGP_data.py \
    --data_dir "./Data" \
    --output_dir "./Output" \
    --snp_start_col 20000 \
    --snp_end_col 26280 \
    --min_allele_freq 0.05 \
    --max_allele_freq 0.95 \
    --max_geno_freq 0.95
