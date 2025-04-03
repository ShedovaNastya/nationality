#!/bin/bash

CSV_PATH='../data/phoneme_energy_test.csv'
CSV_VAL_PATH='../data/phoneme_energy_eval.csv'
MODEL_NAME='t5-base'
PRETRAIN_DIR='../models/voxblink2_samresnet34'
PICKLE_FILE='embeds.pkl'

python3 train.py --csv_path "$CSV_PATH" --csv_val_path "$CSV_VAL_PATH" \
  --model_name "$MODEL_NAME" --wandb_token "$WANDB_TOKEN" \
  --pretrain_dir="$PRETRAIN_DIR" --pickle_path="$PICKLE_FILE"