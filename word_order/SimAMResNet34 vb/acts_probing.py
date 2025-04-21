import argparse
import os
from extract_features import extract_features
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wespeaker


class ActivationDataset(Dataset):
    def __init__(self, activations, labels):
        self.X = self.pad_activations(activations)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def pad_activations(self, activations):
        processed = []
        for act in activations:
            if len(act.shape) == 4:
                act = act.squeeze(0)
            if len(act.shape) == 3:
                act = F.adaptive_avg_pool2d(act, (1, 80))
                act = act.view(act.size(0), -1)
            elif len(act.shape) == 2:
                act = F.adaptive_avg_pool1d(act.unsqueeze(0), 80).squeeze(0)
            processed.append(act.flatten())
        return torch.stack(processed)


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


class WordOrderCls(nn.Module):
    def __init__(self, input_size):
        super(WordOrderCls, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.fc(x))


def get_audio_paths_and_labels(metadata_path):
    df = pd.read_csv(metadata_path)
    audio_paths = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 2].tolist()  
    return audio_paths, labels


def get_activations(model, audio_path, device):
    feats = extract_features(audio_path)
    feats = feats.to(device)

    with torch.no_grad():
        activations, _ = model(feats)

    acts = {
        'file_path': audio_path,
        'act': activations
    }
    return acts


def get_activations_for_layer(model, audio_files, labels, device, layer_name):
    activations = []
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
                activations.append(activation.cpu())
    return activations, labels


def train(train_loader, input_size, layer, device, num_epochs=10):
    model = WordOrderCls(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()

        for X, y in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            X = X.to(device)
            y = y.to(device).float().unsqueeze(1)
            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in tqdm(
                test_loader, desc="Evaluation Progress"):
            X, y = X.to(device), y.to(device)
            preds = model(X).cpu().numpy()
            preds = (np.array(preds) > 0.5).astype(int)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

    return metrics


def plot_metrics(metrics_list, save_path):
    layers = [m[0] for m in metrics_list]
    accuracies = [m[1]["accuracy"] for m in metrics_list]
    f1_scores = [m[1]["f1_score"] for m in metrics_list]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(layers) + 1), accuracies, color='b', label="Accuracy")
    plt.xlabel("Layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy across layers")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(layers) + 1), f1_scores, color='g', label="F1-score")
    plt.xlabel("Layers")
    plt.ylabel("F1-score")
    plt.title("F1-score across layers")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)


def save_metrics(metrics_list, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for layer, metrics in metrics_list:
            f.write(f"{layer}\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        required=True,
        help="Path to wespeaker model pretrain_dir."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="C:/Users/geraz/PycharmProjects/word-order-recognition/word_order/data/metadata.csv",
        help="Path to metadata file"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio"
    )
    parser.add_argument(
        "--models_save_path",
        type=str,
        default="./models",
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
        raise FileNotFoundError(f"Folder {args.pretrain_dir} does not exist.")
    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"File {args.metadata_path} does not exist.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)
    acts_model = GetActivations(model)
    if torch.cuda.is_available():
        acts_model = acts_model.cuda()

    audio_paths, labels = get_audio_paths_and_labels(args.metadata_path)

    from sklearn.model_selection import train_test_split
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_paths,
        labels,
        test_size=1 - args.train_ratio,
        stratify=labels,
        random_state=42
    )

    acts = get_activations(acts_model, train_files[0], device)
    layers = [list(item.keys())[0] for item in acts['act']]

    metrics_list = []

    for layer in layers:
        train_acts, train_labels = get_activations_for_layer(
            acts_model, train_files, train_labels, device, layer)
        test_acts, test_labels = get_activations_for_layer(
            acts_model, test_files, test_labels, device, layer)

        train_dataset = ActivationDataset(train_acts, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        test_dataset = ActivationDataset(test_acts, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        input_size = train_dataset[0][0].shape[0]
        trained_model = train(
            train_loader, input_size, layer, device)
        torch.save(trained_model.state_dict(),
                   f"{args.models_save_path}/{layer}.pth")

        metrics = evaluate(trained_model, test_loader, device)
        print(f"Layer {layer} - Test accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

        metrics_list.append((layer, metrics))
        torch.cuda.empty_cache()

    save_metrics(metrics_list, args.text_save_path)
    plot_metrics(metrics_list, args.visual_save_path)


if __name__ == '__main__':
    main()
