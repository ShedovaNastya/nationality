import argparse
import os
import chromadb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EmbeddingsDataset(Dataset):
    def __init__(self, source_path, split, source_type, collection_name="formants_embeddings"):
        if source_type == "chromadb":
            self.embeddings, self.phonemes, self.labels = self.get_chroma_embeddings(
                source_path, split, collection_name)
        else:
            raise ValueError(f"Invalid source type: {source_type}. Choose 'chromadb'.")
        self.labels = np.array(self.labels)
        self.min_vals = self.labels.min(axis=0)
        self.max_vals = self.labels.max(axis=0)
        self.labels = (self.labels - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)
        print(f"Min labels: {self.min_vals}, Max labels: {self.max_vals}")
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def get_chroma_embeddings(self, source_path, split, collection_name="formants_embeddings"):
        client = chromadb.PersistentClient(path=source_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get(where={"split": split}, include=["embeddings", "metadatas"])
        embeddings = np.array(results['embeddings'], dtype=np.float32)
        phonemes = [item.get('phonemes', '') for item in results['metadatas']]
        labels = [[item['f1'], item['f2'], item['f3']] for item in results['metadatas']]
        return embeddings, phonemes, labels

    def __getitem__(self, idx):
        return self.embeddings[idx], self.phonemes[idx], self.labels[idx]

    def __len__(self):
        return len(self.embeddings)

class FormantPredictor(nn.Module):
    def __init__(self, embedding_dim=256, output_dim=3):
        super(FormantPredictor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(embedding_dim + 768, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, embeddings, phonemes):
        inputs = self.tokenizer(phonemes, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(embeddings.device) for k, v in inputs.items()}
        bert_outputs = self.bert(**inputs).last_hidden_state[:, 0, :]
        combined = torch.cat((embeddings, bert_outputs), dim=-1)
        hidden = torch.relu(self.fc1(combined))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        return combined, output

def train(model, train_loader, test_loader, optimizer, criterion, num_epoch, device, scheduler, patience=20):
    best_mse = float('inf')
    patience_counter = 0
    best_weights_path = os.path.join(args.output_dir, "best_model_weights.pth")
    for epoch in tqdm(range(num_epoch), desc="Training"):
        model.train()
        for embeddings_batch, phonemes_batch, labels_batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}"):
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(embeddings_batch, phonemes_batch)[1]
            loss = criterion(outputs, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics, _, _ = evaluate(model, test_loader, device)
        current_mse = metrics['mse_loss']
        print(f"Epoch {epoch + 1}/{num_epoch}, Test MSE: {current_mse}")
        if current_mse < best_mse:
            best_mse = current_mse
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f"Best model weights saved to {best_weights_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        scheduler.step(current_mse)

def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for embeddings_batch, phonemes_batch, labels_batch in tqdm(
                test_loader, desc="Evaluating"):
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)
            _, outputs = model(embeddings_batch, phonemes_batch)
            loss = nn.MSELoss()(outputs, labels_batch)
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels_batch.cpu().numpy())
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    avg_loss = total_loss / len(test_loader)
    true_labels = true_labels * (test_loader.dataset.max_vals - test_loader.dataset.min_vals) + test_loader.dataset.min_vals
    predictions = predictions * (test_loader.dataset.max_vals - test_loader.dataset.min_vals) + test_loader.dataset.min_vals
    metrics = {"mse_loss": avg_loss}
    for i, formant in enumerate(['f1', 'f2', 'f3']):
        mae = mean_absolute_error(true_labels[:, i], predictions[:, i])
        rmse = np.sqrt(mean_squared_error(true_labels[:, i], predictions[:, i]))
        r2 = r2_score(true_labels[:, i], predictions[:, i])
        metrics[f"{formant}_mae"] = mae
        metrics[f"{formant}_rmse"] = rmse
        metrics[f"{formant}_r2"] = r2
    return metrics, predictions, true_labels

def get_loaders(source_path, source_type):
    train_dataset = EmbeddingsDataset(source_path, split="train", source_type=source_type)
    test_dataset = EmbeddingsDataset(source_path, split="test", source_type=source_type)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return train_loader, test_loader, test_dataset, train_dataset.embeddings.shape[1]

def save_visualization(model, vectors, phonemes, labels, save_path, device):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vectors = torch.FloatTensor(vectors).to(device)
    with torch.no_grad():
        x1, _ = model(vectors, phonemes)
    x1_reduced = TSNE(n_components=2, random_state=42).fit_transform(x1.cpu().numpy())
    labels = np.array(labels)
    ranges = [(1000, 2000), (2000, 3000), (3000, 5000)]
    colors = ['blue', 'green', 'red']
    plt.figure(figsize=(10, 8))
    for i, formant in enumerate(['F1', 'F2', 'F3']):
        plt.subplot(1, 3, i + 1)
        for (low, high), color in zip(ranges, colors):
            indices = (labels[:, i] >= low) & (labels[:, i] < high)
            plt.scatter(
                x1_reduced[indices, 0],
                x1_reduced[indices, 1],
                c=color,
                label=f'{low}-{high} Hz',
                alpha=0.6
            )
        plt.title(f't-SNE for {formant}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_source", type=str, choices=["chromadb"], required=True)
    parser.add_argument("--source_path", type=str, default="C:\\Users\\geraz\\PycharmProjects\\interp_dev_paizula\\intonation_contour\\embeddings")
    parser.add_argument("--output_dir", type=str, default="C:\\Users\\geraz\\PycharmProjects\\interp_dev_paizula\\intonation_contour\\results")
    args = parser.parse_args()
    if not os.path.exists(args.source_path):
        raise FileNotFoundError(f"Folder {args.source_path} does not exists.")
    eval_path = os.path.join(args.output_dir, "formants.txt")
    visual_path = os.path.join(args.output_dir, "formants.png")
    weights_path = os.path.join(args.output_dir, "model_weights.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, test_dataset, input_dim = get_loaders(args.source_path, args.embeddings_source)
    model = FormantPredictor(input_dim, 3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    train(model, train_loader, test_loader, optimizer, criterion, num_epoch=600, device=device, scheduler=scheduler, patience=20)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model_weights.pth")))
    metrics, predictions, true_labels = evaluate(model, test_loader, device)
    save_metrics(metrics, eval_path)
    save_visualization(model, test_dataset.embeddings.numpy(), test_dataset.phonemes, true_labels, visual_path, device)
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

if __name__ == '__main__':
    main()