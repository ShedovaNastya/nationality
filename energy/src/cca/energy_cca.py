import argparse
import os.path
import pickle
import numpy as np
import pandas as pd
import torch
import torchaudio
import wespeaker
import cca_analysis as cca
import cca_score as cca_score

from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from energy.src.cca.extract_features import extract_features


def assign_phoneme_labels(
        phoneme,
        encoder,
        acts,
):
    acts['label'] = encoder.transform(np.array(phoneme).reshape(-1, 1))


def get_activations(
        model,
        wav_path,
        start_time,
        end_time,
        device
):
    feats = extract_features(wav_path, start_time, end_time).to(device)

    with torch.no_grad():
        activations, _ = model(feats)

    acts = {
        'file_path': wav_path,
        'act': activations
    }
    return acts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        required=True,
        help="Path to wespeaker model pretrain_dir."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to audio file"
    )

    parser.add_argument(
        "--visual_save_path",
        type=str,
        default="../../data/result/cca_score.png",
        help="Save path for visualization result"
    )
    parser.add_argument(
        "--text_save_path",
        type=str,
        default="../../data/result/cca_score.txt",
        help="Save path for text result"
    )
    args = parser.parse_args()

    # validate args
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"File {args.csv_path} does not exists.")

    if not os.path.exists(args.pretrain_dir):
        raise FileNotFoundError(f"Folder {args.pretrain_dir} does not exists.")

    # define device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # load model
    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)

    activations_model = cca.GetActivations(model)
    encoder = OneHotEncoder(sparse_output=False)

    df = pd.read_csv(args.csv_path)

    # Получение всех уникальных фонем из датасета
    all_phonemes = df["phoneme"].unique().reshape(-1, 1)

    # Обучение OneHotEncoder
    encoder.fit(all_phonemes)

    cca_scores = []
    layers = None

    for idx, data in tqdm(df.iterrows(), total=len(df), desc="Loading dataset"):
        wav_path = data["wav_path"]
        labels = data["phoneme"]
        start_time = data["start_time"]
        end_time = data["end_time"]

        acts = get_activations(activations_model, wav_path, start_time, end_time, device)
        assign_phoneme_labels(labels, encoder, acts)
        cca_coefs, layers = cca.get_cca(acts, encoder)
        cca_scores.append(cca_coefs)

    label_to_cca = defaultdict(list)

    for item in cca_scores:
        label_to_cca[item['label']].append(item['cca'])

    averaged_cca = {label: np.mean(cca_list, axis=0)
                    for label, cca_list in label_to_cca.items()}

    cca.visualize_cca_score(averaged_cca, args.visual_save_path)
    cca.save_cca(averaged_cca, layers, args.text_save_path)


if __name__ == "__main__":
    main()
