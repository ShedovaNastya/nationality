import os
import chromadb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SAVE_PATH = "embeddings"
COLLECTION_NAME = "speech_embeddings"

class EmbeddingsDataset(Dataset):
    def __init__(self, source_path, split, source_type, collection_name=COLLECTION_NAME):
        self.lb = LabelEncoder()
        if source_type == "chromadb":
            self.embeddings, self.labels = self.get_chroma_embeddings(source_path, split, collection_name)
        else:
            raise ValueError("This code only supports 'chromadb' as source_type")
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def get_chroma_embeddings(self, source_path, split, collection_name):
        client = chromadb.PersistentClient(path=source_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get(where={"split": split}, include=["embeddings", "metadatas"])
        embeddings = np.array(results['embeddings'], dtype=np.float32)
        labels = [item['label'] for item in results['metadatas']]
        labels = self.lb.fit_transform(labels)
        return embeddings, labels

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        return len(self.embeddings)

def get_loaders(source_path, source_type="chromadb"):
    train_dataset = EmbeddingsDataset(source_path, split="train", source_type=source_type)
    test_dataset = EmbeddingsDataset(source_path, split="test", source_type=source_type)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    num_classes = len(train_dataset.lb.classes_)
    return train_loader, test_loader, test_dataset, train_dataset.embeddings.shape[1], num_classes

class SpeakerCls(nn.Module):
    def __init__(self, input_dim=256, num_classes=19):
        super(SpeakerCls, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, num_classes)

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.3)

        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)

    def forward(self, x):
        x1 = self.activation(self.norm1(self.fc1(x)))
        x2 = self.activation(self.norm2(self.fc2(x1)) + x1)
        x2 = self.dropout(x2)
        x3 = self.activation(self.norm3(self.fc3(x2)) + x2)
        output = self.fc4(x3)
        return x3, output


def train(model, train_loader, optimizer, criterion, num_epoch, device):
    for epoch in tqdm(range(num_epoch), desc="Training Progress"):
        model.train()
        for embeddings_batch, labels_batch in train_loader:
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device).long()
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
        for embeddings_batch, labels_batch in tqdm(test_loader, desc="Evaluation Progress"):
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.long()
            _, outputs = model(embeddings_batch)
            _, predicted = torch.max(outputs.cpu(), 1)
            true_labels.extend(labels_batch.numpy())
            pred_labels.extend(predicted.numpy())
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, average="weighted"),
        "recall": recall_score(true_labels, pred_labels, average="weighted"),
        "f1_score": f1_score(true_labels, pred_labels, average="weighted")
    }
    return metrics

def save_metrics(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def save_visualization(model, vectors, labels, test_dataset, save_path, device):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vectors = torch.FloatTensor(vectors).to(device)
    with torch.no_grad():
        x1, _ = model(vectors)
    reducer = TSNE(n_components=2, random_state=42)
    x1_reduced = reducer.fit_transform(x1.detach().cpu().numpy())
    unique_labels = list(set(labels))
    encoded_labels = list(test_dataset.lb.classes_)
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            x1_reduced[indices, 0],
            x1_reduced[indices, 1],
            label=f"Label: {encoded_labels[label]}",
            alpha=0.6
        )
    plt.title("Visualization of embeddings after first layer")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def visualize_raw_embeddings(dataset, save_path):
    embeddings = dataset.embeddings.numpy()
    labels = dataset.labels.numpy()
    reducer = TSNE(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(reduced[indices, 0], reduced[indices, 1], label=f"Class {label}", alpha=0.6)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    train_loader, test_loader, test_dataset, input_dim, num_classes = get_loaders(SAVE_PATH)
    print(f"Input data size: {input_dim}, classes count: {num_classes}")
    model = SpeakerCls(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    visualize_raw_embeddings(test_dataset, "results/raw_embeddings.png")
    train(model, train_loader, optimizer, criterion, num_epoch=100, device=device)
    metrics = evaluate(model, test_loader, device)
    save_metrics(metrics, "results/SpeakerCls.txt")
    print("Metrics:", metrics)
    save_visualization(model, test_dataset.embeddings.numpy(), test_dataset.labels.numpy(), test_dataset, "results/SpeakerCls.png", device)
    os.makedirs("SpeakerCls", exist_ok=True)
    torch.save(model.state_dict(), "SpeakerCls/SpeakerCls.pth")
    print("Save model in SpeakerCls/SpeakerCls.pth")

if __name__ == "__main__":
    main()
