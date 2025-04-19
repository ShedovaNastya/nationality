import argparse
import os

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class EmbeddingsDataset(Dataset):
    def __init__(
        self, source_path, split, source_type, collection_name="order_embeddings"
    ):
        self.lb = LabelEncoder()

        if source_type == "chromadb":
            self.embeddings, self.labels = self.get_chroma_embeddings(
                source_path, split, collection_name
            )

        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def get_chroma_embeddings(self, source_path, split, collection_name):
        client = chromadb.PersistentClient(path=source_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get(
            where={"split": split}, include=["embeddings", "metadatas"]
        )
        embeddings = np.array(results["embeddings"], dtype=np.float32)
        labels = [int(item["is_correct"]) for item in results["metadatas"]]
        labels = self.lb.fit_transform(labels)
        return embeddings, labels

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        return len(self.embeddings)


class OrderCls(nn.Module):
    def __init__(self, input_dim=256, num_classes=2):
        super(OrderCls, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x1, x2


def train(model, train_loader, optimizer, criterion, num_epoch, device):
    for epoch in tqdm(range(num_epoch), desc="Training Progress"):
        model.train()

        for embeddings_batch, labels_batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}"
        ):
            embeddings_batch = embeddings_batch.to(device)

            labels_batch = labels_batch.to(device)

            _, outputs = model(embeddings_batch)
            loss = criterion(outputs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for embeddings_batch, labels_batch in tqdm(
            test_loader, desc="Evaluation Progress"
        ):
            embeddings_batch = embeddings_batch.to(device)
            _, outputs = model(embeddings_batch)

            _, predicted = torch.max(outputs.cpu(), 1)
            true_labels.extend(labels_batch.cpu().numpy())
            pred_labels.extend(predicted.numpy())

    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1_score": f1_score(true_labels, pred_labels),
    }


def get_loaders(source_path, source_type):
    train_dataset = EmbeddingsDataset(source_path, "train", source_type)
    test_dataset = EmbeddingsDataset(source_path, "test", source_type)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, test_dataset, train_dataset.embeddings.shape[1]


def save_visualization(model, vectors, labels, save_path, device):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vectors = torch.FloatTensor(vectors).to(device)
    with torch.no_grad():
        x1, _ = model(vectors)

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(x1.cpu().numpy())

    plt.figure(figsize=(10, 8))
    for label in set(labels):
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            reduced[indices, 0],
            reduced[indices, 1],
            label=f"{'Correct' if label else 'Incorrect'}",
            alpha=0.6,
        )
    plt.title("Order Correctness Embeddings")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def save_metrics(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_source", type=str, choices=["chromadb"], required=True
    )
    parser.add_argument("--source_path", type=str, default="./chroma_db")
    parser.add_argument("--eval_path", type=str, default="./results/metrics.txt")
    parser.add_argument("--visual_path", type=str, default="./results/embeddings.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, test_dataset, input_dim = get_loaders(
        args.source_path, args.embeddings_source
    )
    model = OrderCls(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train(model, train_loader, optimizer, criterion, 100, device)

    model_save_path = os.path.join(os.path.dirname(args.eval_path), "model.pth")
    torch.save(model.state_dict(), model_save_path)

    metrics = evaluate(model, test_loader, device)

    save_metrics(metrics, args.eval_path)
    save_visualization(
        model,
        test_dataset.embeddings.numpy(),
        test_dataset.labels.numpy(),
        args.visual_path,
        device,
    )


if __name__ == "__main__":
    main()
