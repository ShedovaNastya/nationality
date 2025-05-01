import argparse
import os
import sys
sys.path.append(r"C:\Users\User\OneDrive\Рабочий стол\лабык2с2\Проектный практикум\wespeaker\wespeaker")
from extract_features import extract_features
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wespeaker
import json


def load_audio_paths_from_json(json_path, base_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    paths = []
    labels = []
    for item in data:
        full_path = os.path.join(base_dir, item['language'], item['filename'])
        paths.append(full_path)
        labels.append(item['language'])

    return paths, labels


class ActivationDataset(Dataset):
    def __init__(self, activations, labels):
        self.X = self.pad_activations(activations)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def pad_activations(self, activations):
        pooled = []
        for act in activations:
            while act.dim() > 2:
                act = act.mean(dim=-1)
            pooled.append(act.squeeze())
        return torch.stack(pooled)


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
            c_sim = 0
            c_relu = 0
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                for sec_name, sec_layer in layer.named_children():
                    identity = out

                    out = sec_layer.relu(sec_layer.bn1(sec_layer.conv1(out)))
                    c_relu += 1
                    activations.append({f"{name} relu {c_relu}": out})

                    out = sec_layer.bn2(sec_layer.conv2(out))
                    out = sec_layer.SimAM(out)
                    c_sim += 1
                    activations.append({f"{name} SimAM {c_sim}": out})

                    if sec_layer.downsample is not None:
                        identity = sec_layer.downsample(identity)

                    out += identity
                    out = sec_layer.relu(out)
                    c_relu += 1
                    activations.append({f"{name} relu {c_relu}": out})

        out = self.model.model.pooling(out)
        activations.append({"pooling": out})

        if self.model.model.drop:
            out = self.model.model.drop(out)

        out = self.model.model.bottleneck(out)

        return activations, out


class LanguageCls(nn.Module):
    """
    Baseline model class for language classification
    """

    def __init__(self, input_size, num_classes=4):
        super(LanguageCls, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

 
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


def get_activations_for_layer(model, audio_files, device, layer_name, save_dir):
    """
    Gets model activations for a specified layer.
    """
    with torch.no_grad():
        for audio_path in tqdm(
            audio_files,
            desc=f"Extracting {layer_name} activations"
        ):
            feats = extract_features(audio_path).to(device)
            acts, _ = model(feats)

            activation = next((d[layer_name]
                              for d in acts if layer_name in d), None)
            if activation is not None:
                save_path = os.path.join(save_dir, f"{layer_name}_{Path(audio_path).stem}.pt")
                torch.save(activation.cpu(), save_path)
    return


def save_all_layer_activations(model, audio_paths, device, base_save_dir):
    sample_acts = get_activations(model, audio_paths[0], device)
    layers = [list(item.keys())[0] for item in sample_acts['act']]

    for layer in layers:
        layer_save_dir = os.path.join(base_save_dir, layer)
        os.makedirs(layer_save_dir, exist_ok=True)

        get_activations_for_layer(model, audio_paths, device, layer, layer_save_dir)


def load_saved_activations(folder, audio_files, layer):
    activations = []
    for audio_path in audio_files:
        fname = f"{layer}_{Path(audio_path).stem}.pt"
        act_path = os.path.join(folder, layer, fname)
        activation = torch.load(act_path)
        activations.append(activation)
    return activations


def train(train_loader, input_size, layer, device, num_epochs=10):
    """
    Train a model on a train dataset
    """
    model = LanguageCls(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()

        for X, y in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, test_loader, device):
    """
    Evaluates a model on a test dataset.
    Calculates accuracy and f1-score
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in tqdm(
                test_loader, desc="Evaluation Progress"):
            X, y = X.to(device), y.to(device)
            preds = model(X).cpu().numpy()
            preds = np.argmax(preds, axis=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="macro")
    }

    return metrics


def plot_metrics(metrics_list, save_path):
    layers = [m[0] for m in metrics_list]
    accuracies = [m[1]["accuracy"] for m in metrics_list]
    f1_scores = [m[1]["f1_score"] for m in metrics_list]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(layers)+1), accuracies, color='b', label="Accuracy")
    plt.xlabel("Layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy across layers")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(layers)+1), f1_scores, color='g', label="F1-score")
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

    with open(save_path, 'w') as f:
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
        help="Path to train audio files"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./test_audio",
        help="Path to test audio files"
    )
    parser.add_argument(
        "--models_save_path",
        type=str,
        default="./models_test",
        help="Save path for trained models"
    )
    parser.add_argument(
        "--text_save_path",
        type=str,
        default="./result/probing.txt",
        help="Save path for text result"
    )
    parser.add_argument(
        "--visual_save_path",
        type=str,
        default="./result/probing.png",
        help="Save path for visual result"
    )
    args = parser.parse_args()

    if not os.path.exists(args.pretrain_dir):
        raise FileNotFoundError(f"Folder {args.pretrain_dir} does not exists.")
    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Folder {args.train_dir} does not exists.")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Folder {args.test_dir} does not exists.")
    if not os.path.exists(args.models_save_path):
        raise FileNotFoundError(
            f"Folder {args.models_save_path} does not exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)

    acts_model = GetActivations(model)

    train_files, train_labels = load_audio_paths_from_json(
        os.path.join(args.train_dir, "train.json"), args.train_dir
    )
    test_files, test_labels = load_audio_paths_from_json(
        os.path.join(args.test_dir, "test.json"), args.test_dir
    )

    all_labels = train_labels + test_labels

    le = LabelEncoder()
    le.fit(all_labels)

    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)

    if not os.path.exists("D:/folder/probe/saved_activations/train"):
        save_all_layer_activations(acts_model, train_files, device, "D:/folder/probe/saved_activations/train")
    if not os.path.exists("D:/folder/probe/saved_activations/test"):
        save_all_layer_activations(acts_model, test_files, device, "D:/folder/probe/saved_activations/test")
    
    acts = get_activations(acts_model, train_files[0], device)
    layers = [list(item.keys())[0] for item in acts['act']]

    metrics_list = []

    for layer in layers:
        train_acts = load_saved_activations("D:/folder/probe/saved_activations/train", train_files, layer)
        test_acts = load_saved_activations("D:/folder/probe/saved_activations/test", test_files, layer)

        train_dataset = ActivationDataset(train_acts, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        test_dataset = ActivationDataset(test_acts, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        trained_model = train(
            train_loader, train_dataset.X.shape[-1], layer, device)
        torch.save(trained_model.state_dict(),
                   f"{args.models_save_path}/{layer}.pth")
        metrics = evaluate(trained_model, test_loader, device)

        metrics_list.append((layer, metrics))

        torch.cuda.empty_cache()

    save_metrics(metrics_list, args.text_save_path)
    plot_metrics(metrics_list, args.visual_save_path)


if __name__ == '__main__':
    main()