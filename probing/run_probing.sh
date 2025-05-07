#!/bin/bash

TASK="cls"

if [ "$TASK" = "regr" ]; then
    echo "Error: Regression task not implemented." >&2
    exit 1
fi

if [ "$TASK" = "cls" ]; then
    TRAIN_DIR="./train_audio"
    TEST_DIR="./test_audio"
    PRETRAIN_DIR="./pretrain_dir"
    TEXT_SAVE_PATH="./result/probing.txt"
    VISUAL_SAVE_PATH="./result/probing.png"
    CHUNK_SIZE=500

    python probing.py --pretrain_dir "$PRETRAIN_DIR" --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR" \
        --chunk_size "$CHUNK_SIZE" --text_save_path "$TEXT_SAVE_PATH" --visual_save_path "$VISUAL_SAVE_PATH"

    echo "The script has been executed"
fi
