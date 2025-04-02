import argparse
import os

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class EmbeddingsDataset(Dataset):
    """
    Dataset class for loading embeddings
    """
    def __init__(
            self,
            source_path,
            split,
            source_type,
            collection_name="words_embeddings"):
        self.lb = LabelEncoder()

        if source_type == "npy":
            self.embeddings, self.labels = self.get_npy_embeddings(
                source_path, split)
        elif source_type == "chromadb":
            self.embeddings, self.labels = self.get_chroma_embeddings(
                source_path, split, collection_name)
        else:
            raise ValueError(
                f"Invalid source type: {source_type}. "
                "Choose 'npy' or 'chromadb'."
            )

        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def get_npy_embeddings(self, source_path, split):
        source = np.load(os.path.join(source_path, "numpy_embs.npy"), allow_pickle=True)
        source = source[0]

        if split == "train":
            embeddings = np.array([item['embedding'] for item in source['train']])
            labels = [item['label'] for item in source['train']]
        elif split == "test":
            embeddings = np.array([item['embedding'] for item in source['test']])
            labels = [item['label'] for item in source['test']]
        
        label_mapping = {'цвета': 0, 'имена': 1, 'числительные': 2, 'животные': 3}
        labels = [label_mapping[label] for label in labels]

        labels = self.lb.fit_transform(labels)
        return embeddings, labels

    def get_chroma_embeddings(
            self,
            source_path,
            split,
            collection_name="gender_embeddings"):
        """
        Reads embeddings from ChromaDB
        """
        client = chromadb.PersistentClient(path=source_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get(where={"split": split}, include=[
            "embeddings", "metadatas"])
        embeddings = np.array(results['embeddings'], dtype=np.float32)
        labels = [item['label'] for item in results['metadatas']]

        labels = self.lb.fit_transform(labels)
        return embeddings, labels

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        return len(self.embeddings)


class AnimalClassifier(nn.Module):
    """
    Baseline model class for gender classification
    """

    def __init__(self, input_dim=256, num_classes=4):
        super(AnimalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x1, x2


def train(model, train_loader, optimizer, criterion, num_epoch, device):
    """
    Train a model on a train dataset
    """
    for epoch in tqdm(range(num_epoch), desc="Training Progress"):
        model.train()

        for embeddings_batch, labels_batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}"
        ):
            embeddings_batch = embeddings_batch.to(device)

            labels_batch = labels_batch.long()
            _, outputs = model(embeddings_batch)
            loss = criterion(outputs, labels_batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader, device):
    """
    Evaluates a model on a test dataset. Calculates accuracy,
    precision, recall and f1-score
    """
    model.eval()
    total_samples_test = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for embeddings_batch, labels_batch in tqdm(
                test_loader, desc="Evaluation Progress"):
            embeddings_batch = embeddings_batch.to(device)

            labels_batch = labels_batch.long()
            x1, outputs = model(embeddings_batch)

            total_samples_test += 1

            _, predicted = torch.max(outputs.cpu(), 1)
            true_labels.extend(labels_batch.numpy())
            pred_labels.extend(predicted.numpy())

    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, average='macro'),
        "recall": recall_score(true_labels, pred_labels, average='macro'),
        "f1_score": f1_score(true_labels, pred_labels, average='macro')
    }

    return metrics


def get_loaders(source_path, source_type):
    """
    Creates dataloaders for train and test files
    """
    train_dataset = EmbeddingsDataset(
        source_path, split="train", source_type=source_type)
    test_dataset = EmbeddingsDataset(
        source_path, split="test", source_type=source_type)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return (
        train_loader,
        test_loader,
        test_dataset,
        train_dataset.embeddings.shape[1]
    )


def save_visualization(model, vectors, labels, save_path, device, dataset):
    """
    Saves embedding visualization in .png files
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vectors = torch.FloatTensor(vectors).to(device)
    with torch.no_grad():
        x1, predicted = model(vectors)

    reducer = TSNE(n_components=2, random_state=42)
    x1_reduced = reducer.fit_transform(x1.detach().cpu().numpy())
    label_mapping = {0: 'цвета', 1: 'имена', 2: 'числительные', 3: 'животные'}
    unique_labels = list(label_mapping.keys())

    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            x1_reduced[indices, 0],
            x1_reduced[indices, 1],
            label=f"{label_mapping[label]}",
            alpha=0.6
        )

    plt.title("Visualization of embeddings after first layer")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def save_metrics(metrics, save_path):
    """
    Saves computed metrics in .txt file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_source",
        type=str,
        choices=["npy", "chromadb"],
        required=True,
        help="Source for embeddings: npy or chromadb"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="./embeddingz",
        help="Path to npy file or to chromadb collection folder"
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="./result/wordsclassification.txt",
        help="Save path for evaluation results file (txt)"
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default="./result/wordsclassification.png",
        help="Save path for embeddings visualisation"
    )
    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        raise FileNotFoundError(f"Folder {args.source_path} does not exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, test_dataset, input_dim = get_loaders(
        args.source_path, args.embeddings_source
    )
    model = AnimalClassifier(input_dim, 4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train(model, train_loader, optimizer,
          criterion, num_epoch=300, device=device)

    metrics = evaluate(model, test_loader, device)
    save_metrics(metrics, args.eval_path)
    save_visualization(
        model, test_dataset.embeddings.numpy(),
        test_dataset.labels.numpy(), args.visual_path, device=device,
        dataset=test_dataset
    )


if __name__ == '__main__':
    main()
