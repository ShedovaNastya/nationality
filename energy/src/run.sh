#!/bin/bash

CSV_PATH='../data/phoneme_energy-27000.csv'
CSV_VAL_PATH='../data/eval_dataset.csv'
MODEL_NAME='t5-base'
PRETRAIN_DIR='../models/voxblink2_samresnet34'
TSNE_PLOT_PATH='../data/result/tsne.png'

python3 train.py --csv_path "$CSV_PATH" --csv_val_path "$CSV_VAL_PATH" \
  --model_name "$MODEL_NAME" --wandb_token "$WANDB_TOKEN" \
  --pretrain_dir="$PRETRAIN_DIR" --tsne_plot_path="$TSNE_PLOT_PATH"