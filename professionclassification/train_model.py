import argparse
import os
from sklearn.decomposition import PCA
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
    def __init__(self, source_path, split, source_type, collection_name="profession_embeddings"):
        self.lb = LabelEncoder()
        
        if source_type == "npy":
            self.embeddings, self.labels, self.raw_labels = self.get_npy_embeddings(source_path, split)
            # Сохраняем оригинальные метки
            self.label_mapping = {i: lbl for i, lbl in enumerate(self.lb.classes_)}
        elif source_type == "chromadb":
            self.embeddings, self.labels = self.get_chroma_embeddings(source_path, split, collection_name)
            self.raw_labels = self.lb.inverse_transform(self.labels)
            self.label_mapping = {i: lbl for i, lbl in enumerate(self.lb.classes_)}
        else:
            raise ValueError(f"Invalid source type: {source_type}")

        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def get_npy_embeddings(self, source_path, split):
        """
        Reads embeddings from a .npy file
        """
        source = np.load(os.path.join(
            source_path, "numpy_embs.npy"), allow_pickle=True)
        print(f"Total items: {len(source)}")
        print(f"First item keys: {source[0].keys()}")
        print(f"First item split: {source[0]['split']}")
        print(f"First item label: {source[0]['label']}")

        print(f"Загружено {len(source)} элементов из numpy_embs.npy")
        unique_splits = set(item['split'] for item in source)
        print(f"Уникальные значения 'split': {unique_splits}")

        embeddings = np.array([item['embedding']
                            for item in source if item['split'] == split])
        labels_str = [item['label'] for item in source if item['split'] == split]

        # Преобразуем строковые метки в числовые значения
        labels = self.lb.fit_transform(labels_str)
        
        # Возвращаем embeddings, числовые метки и оригинальные строковые метки
        return embeddings, labels, labels_str  # Теперь возвращаем 3 значения

    def get_chroma_embeddings(
            self,
            source_path,
            split,
            collection_name="profession_embeddings"):
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


class ProfClassifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=2):
        super(ProfClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  # Уменьшили размер скрытого слоя
        self.dropout = nn.Dropout(0.7)  # Увеличили dropout
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return x, self.fc2(x)

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
    """
    Creates dataloaders for train and test files
    """
    train_dataset = EmbeddingsDataset(
        source_path, split="train", source_type=source_type)
    test_dataset = EmbeddingsDataset(
        source_path, split="test", source_type=source_type)

    print(f"Размер train_dataset: {len(train_dataset)}")
    print(f"Размер test_dataset: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return (
        train_loader,
        test_loader,
        test_dataset,
        train_dataset.embeddings.shape[1]
    )


def save_visualization(model, vectors, labels, save_path, device, label_mapping=None):
    """Визуализация с использованием t-SNE"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vectors = torch.FloatTensor(vectors).to(device)
    
    with torch.no_grad():
        x1, outputs = model(vectors)
    
    # Используем t-SNE для визуализации
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(labels)-1))
    x1_reduced = tsne.fit_transform(x1.detach().cpu().numpy())
    
    plt.figure(figsize=(12, 10))
    
    # Обновленный способ получения цветовой карты
    cmap = plt.colormaps.get_cmap('tab10')
    unique_labels = np.unique(labels)
    
    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)[0]
        label_name = label_mapping.get(label, f"Class {label}") if label_mapping else f"Class {label}"
        plt.scatter(
            x1_reduced[indices, 0],
            x1_reduced[indices, 1],
            color=cmap(i),
            label=label_name,
            alpha=0.7
        )
    
    plt.title("t-SNE visualization of embeddings")
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        default="./embeddings",
        help="Path to npy file or to chromadb collection folder"
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="./result/professionclassification.txt",
        help="Save path for evaluation results file (txt)"
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default="./result/professionclassification.png",
        help="Save path for embeddings visualisation"
    )
    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        raise FileNotFoundError(f"Folder {args.source_path} does not exists.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Сначала загружаем файлы для проверки дубликатов
    if args.embeddings_source == "npy":
        source = np.load(os.path.join(args.source_path, "numpy_embs.npy"), allow_pickle=True)
        train_files = [item['file_path'] for item in source if item['split'] == 'train']
        test_files = [item['file_path'] for item in source if item['split'] == 'test']
        
        duplicates = set(train_files) & set(test_files)
        if duplicates:
            print(f"НАЙДЕНЫ ДУБЛИКАТЫ МЕЖДУ TRAIN И TEST ({len(duplicates)}):")
            for dup in list(duplicates)[:3]:
                print(dup)
            raise ValueError("Обнаружены общие файлы между train и test наборами!")
    
    # Затем загружаем датасеты
    train_loader, test_loader, test_dataset, input_dim = get_loaders(
        args.source_path, args.embeddings_source
    )
    
    # Анализ эмбеддингов
    train_embs = train_loader.dataset.embeddings.numpy()
    train_labels = train_loader.dataset.labels.numpy()
    
    print("\nАнализ расстояний между эмбеддингами:")
    for label in np.unique(train_labels):
        indices = np.where(train_labels == label)[0]
        mean_dist = np.mean(np.linalg.norm(train_embs[indices][:, None] - train_embs[indices], axis=2))
        print(f"Среднее расстояние для класса {label}: {mean_dist:.4f}")
    
    label_mapping = test_dataset.label_mapping if hasattr(test_dataset, 'label_mapping') else None
    
    # Debug prints
    print("\n=== Data Debug Info ===")
    print(f"Input dimension: {input_dim}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("Train label distribution:", np.bincount(train_loader.dataset.labels.numpy()))
    print("Test label distribution:", np.bincount(test_loader.dataset.labels.numpy()))
    
    model = ProfClassifier(input_dim, num_classes=len(np.unique(train_loader.dataset.labels))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    train(model, train_loader, optimizer, criterion, num_epoch=20, device=device)
    metrics = evaluate(model, test_loader, device)
    
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    save_metrics(metrics, args.eval_path)
    save_visualization(
        model, 
        test_dataset.embeddings.numpy(),
        test_dataset.labels.numpy(),
        args.visual_path,
        device,
        label_mapping=label_mapping
    )

if __name__ == '__main__':
    main()