#!/bin/bash

# Активация виртуального окружения
source .venv/bin/activate

# Путь к модели SimAMResNet34 vb
MODEL_PATH="voxblink2_samresnet100_ft"

# Путь к датасету с фонемами
DATASET_PATH="dataset/phoneme_dataset.pkl"

# Директория для сохранения результатов
RESULT_DIR="LengthOfEachPhoneme/cca/result"
mkdir -p ${RESULT_DIR}

# Ограничение количества образцов для тестирования
MAX_SAMPLES=200

echo "Running CCA analysis for phoneme duration..."

# Запуск анализа CCA для длительности фонем
python LengthOfEachPhoneme/cca/phoneme_duration_analysis.py \
    --pretrain_dir ${MODEL_PATH} \
    --dataset_path ${DATASET_PATH} \
    --max_samples ${MAX_SAMPLES} \
    --visual_save_path ${RESULT_DIR}/phoneme_duration_cca.png \
    --text_save_path ${RESULT_DIR}/phoneme_duration_cca.txt

echo "CCA analysis completed. Results saved to ${RESULT_DIR}/" 