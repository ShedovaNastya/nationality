import argparse
import os
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


class GetActivations(nn.Module):
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
    audio_dir = Path(audio_dir)
    audio_files = []
    folders = set()
    for f in audio_dir.glob('**/*.wav'):
        if f.is_file():
            folder = f.parent.name
            if folder not in folders:
                audio_files.append(f)
                folders.add(folder)
    return audio_files


def get_activations(model, audio_path, device):
    print(f"Processing file: {audio_path}")
    try:
        feats = extract_features(str(audio_path))
        feats = feats.to(device)

        with torch.no_grad():
            activations, _ = model(feats)

        acts = {
            'file_path': audio_path,
            'act': activations
        }
        return acts
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def assign_labels(acts, audio_files, encoder):
    labels = set()
    for audio_path in audio_files:
        class_name = Path(audio_path).parent.name
        labels.add(class_name)
    encoder.fit(np.array(list(labels)).reshape(-1, 1))

    class_name = acts['file_path'].parent.name
    acts['label'] = encoder.transform([[class_name]])


def get_cca(acts_list, encoder):
    cca_coefs_list = []
    layers = [list(item.keys())[0] for item in acts_list[0]['act']]

    for _ in range(len(layers)):
        cca_coefs_list.append([])

    for acts in acts_list:
        oh_label = acts['label']
        for layer_idx, layer in enumerate(layers):
            act_value = acts['act'][layer_idx][layer]
            act_new = act_value.cpu().numpy()
            if len(act_new.shape) == 4:
                act_new = act_new.squeeze(0).mean(axis=2)
            elif len(act_new.shape) == 2:
                act_new = act_new.T
            else:
                print(f"Bad shape for layer {layer}: {act_new.shape}")
                cca_coefs_list[layer_idx].append(0.0)
                continue

            labels_repeated = np.tile(oh_label.flatten(), (act_new.shape[1], 1)).T

            try:
                cca = CCA(act_new, labels_repeated)
                cca_results = cca.get_cca_parameters(
                    epsilon_x=1e-4,
                    epsilon_y=1e-4,
                    verbose=False
                )
                cca_coefs_list[layer_idx].append(np.mean(cca_results["cca_coef1"]))
            except Exception as e:
                print(f"Error in CCA for layer {layer}: {e}")
                cca_coefs_list[layer_idx].append(0.0)

    cca_coefs = [np.mean(coefs) for coefs in cca_coefs_list]
    return {'label': 'class', 'cca': cca_coefs}, layers


def visualize_cca_score(cca_coefs, save_path):
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for label, coef in cca.items():
            f.write(f"{label}\n")
            for i in range(len(coef)):
                f.write(f"{layers[i]}: {coef[i]}\n")


def main():
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
        default="./data/libri_tts_r_data_test_clean/LibriTTS_R/test-clean",
        help="Path to audio file"
    )
    parser.add_argument(
        "--visual_save_path",
        type=str,
        default="./intonation_contour/cca_results/cca_score.png",
        help="Save path for visualization result"
    )
    parser.add_argument(
        "--text_save_path",
        type=str,
        default="./intonation_contour/cca_results/cca_score.txt",
        help="Save path for text result"
    )
    args = parser.parse_args()

    if not os.path.exists(args.pretrain_dir):
        raise FileNotFoundError(f"Folder {args.pretrain_dir} does not exists.")
    if not os.path.exists(args.audio_dir):
        raise FileNotFoundError(f"Folder {args.audio_dir} does not exists.")

    device = torch.device("cpu")

    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)

    acts_model = GetActivations(model)

    audio_files = get_audio_path(args.audio_dir)
    encoder = OneHotEncoder(sparse_output=False)

    acts_list = []
    for audio_path in tqdm(
        audio_files, desc="Activations computing process"
    ):
        acts = get_activations(acts_model, audio_path, device)
        if acts is None:
            continue
        assign_labels(acts, audio_files, encoder)
        acts_list.append(acts)
        del acts

    if not acts_list:
        raise ValueError("No valid activations computed. Check audio files.")

    cca_coefs, layers = get_cca(acts_list, encoder)

    label_to_cca = defaultdict(list)
    label_to_cca[cca_coefs['label']].append(cca_coefs['cca'])

    averaged_cca = {label: np.mean(cca_list, axis=0)
                    for label, cca_list in label_to_cca.items()}

    visualize_cca_score(averaged_cca, args.visual_save_path)
    save_cca(averaged_cca, layers, args.text_save_path)


if __name__ == '__main__':
    main()