#!/bin/bash

CSV_PATH='../data/phoneme_energy-27000.csv'
CSV_VAL_PATH='../data/eval_dataset.csv'
MODEL_NAME='t5-base'
WANDB_TOKEN=''

python3 train.py --csv_path "$CSV_PATH" --csv_eval_path "$CSV_VAL_PATH" \
  --model_name "$MODEL_NAME" --wandb_token "$WANDB_TOKEN"
#python cca_analysis.py --pretrain_dir "$PRETRAIN_DIR" --audio_dir "$AUDIO_DIR" \
 #--visual_save_path "$VISUAL_PATH" --text_save_path "$TEXT_PATH"
 #
 #echo "Script has been executed."
