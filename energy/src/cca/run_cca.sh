#!/bin/zsh

export PYTHONPATH=$PYTHONPATH:$(pwd)/../../..

CSV_PATH="../../data/phoneme_energy_eval_2.csv"
PRETRAIN_DIR="../../models/voxblink2_samresnet34"
VISUAL_SAVE_PATH="../../data/result/cca/voxblink2_samresnet34/cca_score.png"
TEXT_SAVE_PATH="../../data/result/cca/voxblink2_samresnet34/cca_score.txt"

python3.11 energy_cca.py \
    --csv_path $CSV_PATH \
    --visual_save_path $VISUAL_SAVE_PATH \
    --text_save_path $TEXT_SAVE_PATH \
    --pretrain_dir $PRETRAIN_DIR