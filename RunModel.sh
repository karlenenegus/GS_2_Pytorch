#!/bin/bash

# Run the training script with arguments
python Kfold_train.py \
    --folds 10 \
    --epochs 400 \
    --batch_size 17 \
    --PCs 1691 \
    --model_type "DNNGP" \
    --lr 0.00001 \
    --momentum 0.5 \
    --weight_decay 0.00001 \
    --input_file "./Output/DNNGP_input.pt"
