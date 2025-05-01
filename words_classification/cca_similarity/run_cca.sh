#!/bin/bash

PRETRAIN_DIR="./pretrain_dir"
AUDIO_DIR="./audio"
VISUAL_PATH="./result/cca_score.png"
TEXT_PATH="./result/cca_score.txt"


python cca_analysis.py --pretrain_dir "$PRETRAIN_DIR" --audio_dir "$AUDIO_DIR" \
--visual_save_path "$VISUAL_PATH" --text_save_path "$TEXT_PATH"

echo "Script has been executed."