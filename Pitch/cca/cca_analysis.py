import argparse
import os
import sys
from extract_features import extract_features
from cca_score import CCA
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from tqdm import tqdm
import wespeaker
import csv


class GetActivations(nn.Module):
    """
    Class for getting activations from a model.
    """

    def __init__(self, model):
        super(GetActivations, self).__init__()
        self.model = model

    def forward(self, x):
        out = x.permute(0, 2, 1)
        activations = []
        model_front = self.model.model.front

        x = out.unsqueeze(dim=1)

        out = model_front.relu(model_front.bn1(model_front.conv1(x)))
        activations.append({"first relu": out})

        for name, layer in model_front.named_children():
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                for sec_name, sec_layer in layer.named_children():
                    identity = out

                    out = sec_layer.relu(sec_layer.bn1(sec_layer.conv1(out)))
                    activations.append({f"{name} relu": out})

                    out = sec_layer.bn2(sec_layer.conv2(out))
                    out = sec_layer.SimAM(out)
                    activations.append({"SimAM": out})

                    if sec_layer.downsample is not None:
                        identity = sec_layer.downsample(identity)

                    out += identity
                    out = sec_layer.relu(out)
                    activations.append({f"{name} relu": out})

        out = self.model.model.pooling(out)
        activations.append({"pooling": out})

        if self.model.model.drop:
            out = self.model.model.drop(out)

        out = self.model.model.bottleneck(out)

        return activations, out


def get_audio_path(audio_dir):
    """
    Recursively finds all audio files in the specified directory.
    """
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob('**/*.wav')) + list(
        audio_dir.glob('**/*.mp3'))

    return audio_files


def get_activations(model, audio_path, device):
    """
    Gets model activations.
    """
    feats = extract_features(audio_path)
    feats = feats.to(device)

    with torch.no_grad():
        activations, _ = model(feats)

    acts = {
        'file_path': audio_path,
        'act': activations
    }
    return acts


def load_pitch_labels(csv_path):
    """
    Loads pitch values from CSV file.
    """
    pitch_labels = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['Файл']
            pitch_values = row['Средние питчи (Hz)'].split()

            # Process pitch values and replace N/A with NaN for further processing
            processed_values = []
            for val in pitch_values:
                if val == 'N/A':
                    processed_values.append(np.nan)
                else:
                    try:
                        processed_values.append(float(val))
                    except ValueError:
                        processed_values.append(np.nan)

            # Replace NaN values with mean of non-NaN values
            pitch_array = np.array(processed_values)
            non_nan_values = pitch_array[~np.isnan(pitch_array)]

            if len(non_nan_values) > 0:
                mean_pitch = np.mean(non_nan_values)
                pitch_array = np.nan_to_num(pitch_array, nan=mean_pitch)
            else:
                # If all values are NaN, use a default value (e.g., 0)
                pitch_array = np.zeros_like(pitch_array)

            pitch_labels[filename] = pitch_array

    return pitch_labels


def assign_pitch_labels(acts, pitch_labels):
    """
    Assigns pitch labels to activations.
    For pitch values, we use the numerical values directly.
    """
    file_name = Path(acts['file_path']).name

    if file_name in pitch_labels:
        acts['label'] = pitch_labels[file_name]
    else:
        # If no pitch data found for this file, use zeros
        print(f"Warning: No pitch data found for {file_name}")
        acts['label'] = np.zeros(1)  # Default empty label


def get_cca_with_pitch(acts):
    """
    Computes CCA similarity between activation and pitch values.
    """
    cca_coefs = []
    pitch_values = acts['label']
    file_name = Path(acts['file_path']).name
    layers = [list(item.keys())[0] for item in acts['act']]

    # Ensure pitch values are numpy array
    if not isinstance(pitch_values, np.ndarray):
        pitch_values = np.array(pitch_values)

    # Reshape pitch values to 2D array if needed
    if len(pitch_values.shape) == 1:
        pitch_values = pitch_values.reshape(1, -1)

    for act in acts['act']:
        for act_value in act.values():
            act_new = act_value.cpu().view(64, -1).numpy()

            # Prepare the pitch values to match activation dimensions
            # We might need to repeat or pad the pitch values
            n_repeats = max(1, act_new.shape[1] // pitch_values.shape[1])
            pitch_repeated = np.tile(pitch_values, (2, n_repeats))

            # Truncate or pad to match exactly
            if pitch_repeated.shape[1] > act_new.shape[1]:
                pitch_repeated = pitch_repeated[:, :act_new.shape[1]]
            elif pitch_repeated.shape[1] < act_new.shape[1]:
                padding = np.zeros((pitch_repeated.shape[0], act_new.shape[1] - pitch_repeated.shape[1]))
                pitch_repeated = np.hstack((pitch_repeated, padding))

            # Compute CCA
            cca = CCA(act_new, pitch_repeated)
            cca_results = cca.get_cca_parameters(
                epsilon_x=1e-4,
                epsilon_y=1e-4,
                verbose=False
            )

            cca_coefs.append(np.mean(cca_results["cca_coef1"]))

    return {'file': file_name, 'cca': cca_coefs}, layers


def visualize_cca_score(cca_coefs, save_path):
    """
    Saves CCA similarity visualization in .png files.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 4))
    for label, coef in cca_coefs.items():
        plt.plot(range(1, len(coef) + 1), coef, label=label)

    plt.title("Visualization of CCA Similarity")
    plt.xlabel("Layer number")
    plt.ylabel("CCA similarity")
    plt.legend()

    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def save_cca(cca, layers, save_path):
    """
    Saves computed CCA similarity in .txt file for each layer.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for label, coef in cca.items():
            f.write(f"{label}\n")
            for i in range(len(coef)):
                f.write(f"{layers[i]}: {coef[i]}\n")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        required=True,
        help="Path to wespeaker model pretrain_dir."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="./audio",
        help="Path to audio file"
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="Path to CSV file with pitch labels"
    )
    parser.add_argument(
        "--visual_save_path",
        type=str,
        default="./result/cca_score.png",
        help="Save path for visualization result"
    )
    parser.add_argument(
        "--text_save_path",
        type=str,
        default="./result/cca_score.txt",
        help="Save path for text result"
    )
    args = parser.parse_args()

    if not os.path.exists(args.pretrain_dir):
        raise FileNotFoundError(f"Folder {args.pretrain_dir} does not exists.")
    if not os.path.exists(args.audio_dir):
        raise FileNotFoundError(f"Folder {args.audio_dir} does not exists.")
    if not os.path.exists(args.labels_csv):
        raise FileNotFoundError(f"CSV file {args.labels_csv} does not exist.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)

    acts_model = GetActivations(model)

    audio_files = get_audio_path(args.audio_dir)

    pitch_labels = load_pitch_labels(args.labels_csv)

    cca_score = []
    layers = None

    for audio_path in tqdm(
            audio_files, desc="Activations computing process"
    ):
        acts = get_activations(acts_model, audio_path, device)
        assign_pitch_labels(acts, pitch_labels)

        if np.all(acts['label'] == 0):
            # Skip files with no pitch data
            continue

        cca_coefs, layers = get_cca_with_pitch(acts)
        cca_score.append(cca_coefs)

    label_to_cca = defaultdict(list)

    for item in cca_score:
        label_to_cca[item['label']].append(item['cca'])

    averaged_cca = {label: np.mean(cca_list, axis=0)
                    for label, cca_list in label_to_cca.items()}

    visualize_cca_score(averaged_cca, args.visual_save_path)
    save_cca(averaged_cca, layers, args.text_save_path)


if __name__ == '__main__':
    main()