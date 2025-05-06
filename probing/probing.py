import argparse
from base_dataset import BaseDataset
from extract_features import extract_features
import os
import shutil
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wespeaker


class ActivationDataset(BaseDataset):
    def __init__(self, activations, labels):
        self.audio_data = self.prepare_data(activations)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def prepare_data(self, activations):
        activations = [act.clone() for act in activations]
        max_len = max(act.shape[-1] for act in activations)

        for i in range(len(activations)):
            pad_size = max_len - activations[i].shape[-1]
            activations[i] = torch.nn.functional.pad(
                activations[i], (0, pad_size), value=0.0)
            if len(activations[i].shape) != 2:
                activations[i] = activations[i].view(
                    activations[i].size(0), -1)

        return torch.stack(activations).squeeze(1)


def prepare_chunks(train_files, chunk_size, random_state=42):
    """
    Splits the data into stratified chunks.
    """
    n_chunks = max(2, len(train_files) // chunk_size)
    file_paths = np.array(train_files)
    file_labels = np.array([Path(f).parent.name for f in train_files])

    skf = StratifiedKFold(n_splits=n_chunks, shuffle=True,
                          random_state=random_state)
    return skf, file_paths, file_labels


class GetActivations(nn.Module):
    """
    Class for getting activations from a model.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.saved_out = None

    def save_identity(self, file_name):
        folder = Path("tmp_identity")
        folder.mkdir(exist_ok=True)

        file_path = folder / file_name
        torch.save(self.saved_out, file_path)

    def delete_identity(self):
        if os.path.exists("tmp_identity"):
            shutil.rmtree("tmp_identity")

    def forward(
            self, x, target_layer, from_activation=False, identity_file=None
    ):
        activations = {}
        model_front = self.model.model.front
        out = x
        if not from_activation:
            out = x.permute(0, 2, 1).unsqueeze(dim=1)
            out = model_front.relu(model_front.bn1(model_front.conv1(out)))
            self.saved_out = out.clone()
            if identity_file:
                self.save_identity(identity_file)
            if "first relu" == target_layer:
                activations["first relu"] = out
                return activations, out
        elif from_activation:
            if identity_file and os.path.exists(
                "tmp_identity/" + identity_file
            ):
                self.saved_out = torch.load(
                    "tmp_identity/" + identity_file, map_location=x.device)
            out = x

        for name, layer in model_front.named_children():
            c_sim = 0
            c_relu = 0

            for block_idx, block in layer.named_children():
                identity = self.saved_out

                c_relu += 1
                if f"{name} relu {c_relu}" == target_layer:
                    out = block.relu(block.bn1(block.conv1(out)))
                    activations[f"{name} relu {c_relu}"] = out
                    return activations, out

                c_sim += 1
                if f"{name} SimAM {c_sim}" == target_layer:
                    out = block.bn2(block.conv2(out))
                    out = block.SimAM(out)
                    activations[f"{name} SimAM {c_sim}"] = out
                    return activations, out

                c_relu += 1
                if f"{name} relu {c_relu}" == target_layer:
                    if block.downsample is not None:
                        identity = block.downsample(identity)
                    out += identity
                    out = block.relu(out)
                    self.saved_out = out.clone()
                    if identity_file:
                        self.save_identity(identity_file)
                    activations[f"{name} relu {c_relu}"] = out
                    return activations, out

        if "pooling" == target_layer:
            out = self.model.model.pooling(out)
            activations["pooling"] = out
            return activations, out

        if self.model.model.drop:
            out = self.model.model.drop(out)

        return activations, out


class GenderCls(nn.Module):
    """
    Baseline model class for gender classification
    """

    def __init__(self, input_size, num_classes=1):
        super(GenderCls, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.fc(x))


def get_layers(model):
    """
    Returns SimAM ResNet's layers
    """
    layers = []

    model_front = model.model.front
    layers.append("first relu")

    for name, layer in model_front.named_children():
        c_relu = 0
        c_sim = 0
        if name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for sec_name, sec_layer in layer.named_children():
                c_relu += 1
                layers.append(f"{name} relu {c_relu}")
                c_sim += 1
                layers.append(f"{name} SimAM {c_sim}")
                c_relu += 1
                layers.append(f"{name} relu {c_relu}")
    layers.append("pooling")

    return layers


def get_audio_path(audio_dir):
    """
    Recursively finds all audio files in the specified directory.
    """
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob('**/*.wav')) + list(
        audio_dir.glob('**/*.mp3'))
    return audio_files


def get_activations(model, audio_files, device, chunk_num, layer):
    """
    Gets model activations for a specified layer.
    """
    label_encoder = LabelEncoder()
    labels = [Path(f).parent.name for f in audio_files]
    labels = label_encoder.fit_transform(labels)

    activations = []
    with torch.no_grad():
        for i, audio_path in enumerate(tqdm(
            audio_files, desc="Extracting activations"
        )):
            feats = extract_features(audio_path).to(device)
            acts, _ = model(
                feats, layer,
                identity_file=f"identity_{chunk_num}_{i}.pt")
            activations.append(acts[layer].cpu())
    return activations, labels


def resume_test_layer(metrics_path):
    """
    Defines last test layer
    """
    if not Path(metrics_path).exists():
        return None
    with open(metrics_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in reversed(lines):
        if ':' not in line:
            return line
    return None


def save_tmp(data, file_name):
    """
    Saves temp activations
    """
    folder = Path("tmp")
    folder.mkdir(exist_ok=True)

    file_path = folder / file_name
    torch.save(data, file_path)


def delete_tmp():
    """
    Deletes temp activations
    """
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")


def train_model(
        train_loader,
        input_dim,
        device,
        num_epoch=3,
        existing_model=None
):
    """
    Train a model on a train dataset.
    """
    model = existing_model or GenderCls(input_dim).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for epoch in tqdm(range(num_epoch), desc="Training Progress"):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model


def evaluate(layer, y_pred, y_true):
    """
    Evaluates a model on a test dataset.
    Calculates accuracy and f1-score
    """
    y_pred_labels = (y_pred >= 0.5).astype(int).squeeze(1)

    acc = accuracy_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)

    return (layer, {"accuracy": acc, "f1_score": f1})


def read_metrics(file_path):
    """
    Reads test metrics from .txt file
    """
    metrics_list = []
    current_layer = None
    current_metrics = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_layer is not None and current_metrics:
                    metrics_list.append((current_layer, current_metrics))
                    current_layer = None
                    current_metrics = {}
                continue

            if ':' not in line:
                if current_layer is not None and current_metrics:
                    metrics_list.append((current_layer, current_metrics))
                    current_metrics = {}
                current_layer = line
            else:
                key, value = line.split(':', 1)
                current_metrics[key.strip()] = float(value.strip())

    if current_layer is not None and current_metrics:
        metrics_list.append((current_layer, current_metrics))

    return metrics_list


def plot_metrics(metrics_list, save_path):
    """
    Saves metrics visualization in .png file
    """
    layers = [m[0] for m in metrics_list]
    accuracies = [m[1]["accuracy"] for m in metrics_list]
    f1_scores = [m[1]["f1_score"] for m in metrics_list]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(layers, accuracies, color='b', label="Accuracy")
    plt.xticks(rotation=90, fontsize=6)
    plt.xlabel("Layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy across layers")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(layers, f1_scores, color='g', label="F1-score")
    plt.xticks(rotation=90, fontsize=6)
    plt.xlabel("Layers")
    plt.ylabel("F1-score")
    plt.title("F1-score across layers")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)


def save_metrics(metrics_list, save_path):
    """
    Saves computed metrics in .txt file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'a') as f:
        for layer, metrics in metrics_list:
            f.write(f"{layer}\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


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
        "--train_dir",
        type=str,
        default="./train_audio",
        help="Path to train audio files."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./test_audio",
        help="Path to test audio files."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of file chunks per epoch."
    )
    parser.add_argument(
        "--text_save_path",
        type=str,
        default="./result/probing.txt",
        help="Save path for text result."
    )
    parser.add_argument(
        "--visual_save_path",
        type=str,
        default="./result/probing.png",
        help="Save path for visual result."
    )
    args = parser.parse_args()

    if not os.path.exists(args.pretrain_dir):
        raise FileNotFoundError(f"Folder {args.pretrain_dir} does not exists.")
    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Folder {args.train_dir} does not exists.")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Folder {args.test_dir} does not exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)

    acts_model = GetActivations(model)

    train_files = get_audio_path(args.train_dir)
    test_files = get_audio_path(args.test_dir)

    resume_file = "last_layer.json"
    resume_test_file = args.text_save_path

    layers = get_layers(model)

    if Path(resume_file).exists():
        with open(resume_file, "r") as f:
            resume_layer = json.load(f).get("last_layer")
    else:
        resume_layer = None

    if resume_layer is not None and resume_layer == layers[-1]:
        print(
            "Last training layer already completed. "
            "Skipping training and going to test."
        )
        layers = []
    elif resume_layer is not None:
        try:
            resume_index = layers.index(resume_layer)
            layers = layers[resume_index + 1:]
        except ValueError:
            raise ValueError(
                f"Resume layer {resume_layer} not found in model.")

    if layers:
        skf, file_paths, file_labels = prepare_chunks(
            train_files, args.chunk_size)

        for layer in layers:
            print(f"Processing layer: {layer}")
            probing_model = None
            for i, (_, chunk_idx) in enumerate(
                skf.split(file_paths, file_labels)
            ):
                chunk = file_paths[chunk_idx]
                if not os.path.exists(f"tmp/tmp_acts_{i}.pt"):
                    train_acts, train_labels = get_activations(
                        acts_model, chunk, device, i, layer)
                    train_dataset = ActivationDataset(train_acts, train_labels)
                    train_loader = DataLoader(
                        train_dataset, batch_size=32, shuffle=True)
                else:
                    train_acts = []
                    acts_list = torch.load(f"tmp/tmp_acts_{i}.pt")
                    train_labels = torch.load(f"tmp/tmp_labels_{i}.pt")
                    for num, act in enumerate(acts_list):
                        with torch.no_grad():
                            act = act.to(device)
                            acts, _ = acts_model(
                                act, layer, True,
                                identity_file=f"identity_{i}_{num}.pt")
                            train_acts.append(acts[layer])
                    train_dataset = ActivationDataset(train_acts, train_labels)
                    train_loader = DataLoader(
                        train_dataset, batch_size=32, shuffle=True)

                if probing_model is None:
                    probing_model = train_model(
                        train_loader,
                        input_dim=train_dataset.audio_data.shape[-1],
                        device=device
                    )
                else:
                    probing_model = train_model(
                        train_loader,
                        input_dim=train_dataset.audio_data.shape[-1],
                        device=device,
                        existing_model=probing_model
                    )
                save_tmp(train_acts, f"tmp_acts_{i}.pt")
                save_tmp(train_labels, f"tmp_labels_{i}.pt")

            torch.save(probing_model.state_dict(), f"./models/{layer}.pth")
            with open(resume_file, "w") as f:
                json.dump({"last_layer": layer}, f)

    acts_model.delete_identity()
    delete_tmp()

    print("Testing")
    layers = get_layers(model)
    resume_test = resume_test_layer(resume_test_file)
    if resume_test is not None:
        if resume_test == layers[-1]:
            print(
                "Last test layer already evaluated. Exiting and visualizing."
            )
            plot_metrics(read_metrics(args.text_save_path),
                         args.visual_save_path)
            return
        try:
            resume_index = layers.index(resume_test)
            layers = layers[resume_index + 1:]
        except ValueError:
            raise ValueError(f"Resume test layer {resume_test} not found.")

    skf, file_paths, file_labels = prepare_chunks(test_files, args.chunk_size)

    for layer in layers:
        print(f"Processing layer: {layer}")
        all_preds = []
        all_labels = []

        for i, (_, chunk_idx) in enumerate(skf.split(file_paths, file_labels)):
            chunk = file_paths[chunk_idx]
            if not os.path.exists(f"tmp/tmp_acts_{i}.pt"):
                test_acts, test_labels = get_activations(
                    acts_model, chunk, device, i, layer)
                dataset = ActivationDataset(test_acts, test_labels)
                loader = DataLoader(dataset, batch_size=32, shuffle=False)
            else:
                test_acts = []
                acts_list = torch.load(f"tmp/tmp_acts_{i}.pt")
                test_labels = torch.load(f"tmp/tmp_labels_{i}.pt")
                for num, act in enumerate(acts_list):
                    with torch.no_grad():
                        act = act.to(device)
                        acts, _ = acts_model(
                            act, layer, True,
                            identity_file=f"identity_{i}_{num}.pt"
                        )
                        test_acts.append(acts[layer])

                dataset = ActivationDataset(test_acts, test_labels)
                loader = DataLoader(dataset, batch_size=32, shuffle=True)

            save_tmp(test_acts, f"tmp_acts_{i}.pt")
            save_tmp(test_labels, f"tmp_labels_{i}.pt")
            probing_model = GenderCls(dataset.audio_data.shape[-1]).to(device)
            probing_model.load_state_dict(torch.load(
                f"./models/{layer}.pth", weights_only=True))
            probing_model.eval()

            y_pred_chunk, y_true_chunk = [], []
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    X_batch = X_batch.to(device)
                    outputs = probing_model(X_batch).cpu()
                    y_pred_chunk.extend(outputs.numpy())
                    y_true_chunk.extend(y_batch.numpy())

            all_preds.extend(y_pred_chunk)
            all_labels.extend(y_true_chunk)

            del test_acts, test_labels, dataset, loader, probing_model
            torch.cuda.empty_cache()

        y_pred = np.array(all_preds)
        y_true = np.array(all_labels)
        metrics = evaluate(layer, y_pred, y_true)

        save_metrics([metrics], args.text_save_path)

    acts_model.delete_identity()
    delete_tmp()

    plot_metrics(read_metrics(args.text_save_path), args.visual_save_path)


if __name__ == '__main__':
    main()
