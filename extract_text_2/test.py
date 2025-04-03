import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, PreTrainedModel, \
    PretrainedConfig
import chromadb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import evaluate
import os
from sklearn.manifold import TSNE
import random

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class ASRDataset(Dataset):
    def __init__(self, chroma_db_path="embeddings", split="train", max_length=128, sample_fraction=1.0):
        self.split = split
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.sample_fraction = sample_fraction

        try:
            self.collection = self.client.get_collection("speech_embeddings")
        except ValueError as e:
            raise ValueError(f"Collection 'speech_embeddings' not found into {chroma_db_path}") from e

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
        self.max_length = max_length

        print(f"\nLoading {split} data from ChromaDB...")
        self.data = []
        self.labels = []
        self.embeddings = []

        results = self.collection.get(
            where={"split": split},
            include=["embeddings", "metadatas"]
        )

        if not results or "embeddings" not in results or "metadatas" not in results:
            raise ValueError(f"failed to load data for split={split}")

        all_data = []
        for emb, meta in zip(results["embeddings"], results["metadatas"]):
            if not all(key in meta for key in ["label", "file_path"]):
                continue
            all_data.append({
                "embedding": np.array(emb),
                "label": meta["label"],
                "file_path": meta["file_path"]
            })

        if not all_data:
            raise ValueError(f"data does not exists for split={split} after filtration")

        if self.sample_fraction < 1.0:
            sample_size = int(len(all_data) * self.sample_fraction)
            all_data = random.sample(all_data, sample_size)

        for item in all_data:
            self.data.append(item)
            self.labels.append(item["label"])
            self.embeddings.append(item["embedding"])

        print(f"Loaded {len(self.data)} {split} samples (sampled with fraction {self.sample_fraction})")

        self.embeddings = torch.FloatTensor(np.array(self.embeddings))
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["label"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "audio_embeddings": torch.tensor(item["embedding"], dtype=torch.float32),
            "attention_mask": torch.ones(1),
            "labels": encoding.input_ids.squeeze()
        }


class ASRDataCollator:
    def __call__(self, features):
        audio_embeddings = torch.stack([f["audio_embeddings"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        return {
            "audio_embeddings": audio_embeddings,
            "labels": labels
        }


class ASRModelConfig(PretrainedConfig):
    model_type = "asr_model"

    def __init__(self, t5_model_name="t5-base", audio_emb_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.t5_model_name = t5_model_name
        self.audio_emb_dim = audio_emb_dim


class ASRModel(PreTrainedModel):
    config_class = ASRModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.t5 = T5ForConditionalGeneration.from_pretrained(config.t5_model_name)
        t5_config = self.t5.config
        self.audio_projector = torch.nn.Sequential(
            torch.nn.Linear(config.audio_emb_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, t5_config.d_model)
        )
        self.t5.config.tie_word_embeddings = False

    def forward(self, audio_embeddings, attention_mask=None, labels=None):
        projected = self.audio_projector(audio_embeddings)
        inputs_embeds = projected.unsqueeze(1)
        return self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )


def save_visualization(model, vectors, labels, test_dataset, save_path, device):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vectors = torch.FloatTensor(vectors).to(device)
    with torch.no_grad():
        x1 = model.audio_projector(vectors)

    reducer = TSNE(n_components=2, random_state=42)
    x1_reduced = reducer.fit_transform(x1.detach().cpu().numpy())

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        plt.scatter(
            x1_reduced[indices, 0],
            x1_reduced[indices, 1],
            label=f"Label: {label}",
            alpha=0.6
        )

    plt.title("Visualization of projected embeddings")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    pred_ids = np.argmax(predictions, axis=-1)
    label_ids = labels

    mask = label_ids != -100
    pred_ids = pred_ids[mask]
    label_ids = label_ids[mask]

    accuracy = accuracy_score(label_ids.flatten(), pred_ids.flatten())
    precision = precision_score(label_ids.flatten(), pred_ids.flatten(), average='weighted', zero_division=0)
    recall = recall_score(label_ids.flatten(), pred_ids.flatten(), average='weighted', zero_division=0)
    f1 = f1_score(label_ids.flatten(), pred_ids.flatten(), average='weighted', zero_division=0)

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    pred_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in pred_ids]
    label_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in label_ids]

    filtered_pairs = [(p, l) for p, l in zip(pred_texts, label_texts) if p and l]
    if not filtered_pairs:
        wer = float('inf')
    else:
        filtered_pred_texts, filtered_label_texts = zip(*filtered_pairs)
        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=filtered_pred_texts, references=filtered_label_texts)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "wer": wer
    }

def test_and_visualize():
    chroma_db_path = "embeddings"
    max_length = 8
    batch_size = 16
    model_path = "./asr_wespeaker_final"

    os.makedirs("visualizations", exist_ok=True)

    print("\nPreparing test dataset...")
    test_dataset = ASRDataset(chroma_db_path, split="test", max_length=max_length, sample_fraction=0.3)

    print("Loading model and tokenizer...")
    config = ASRModelConfig(t5_model_name="t5-base", audio_emb_dim=256)
    model = ASRModel.from_pretrained(model_path, config=config).to(device)

    print("Visualizing projected embeddings...")
    save_visualization(
        model,
        test_dataset.embeddings.numpy(),
        test_dataset.labels,
        test_dataset,
        "visualizations/projected_embeddings_visualization.png",
        device
    )

    test_args = TrainingArguments(
        output_dir="./asr_wespeaker_test",
        per_device_eval_batch_size=batch_size,
        logging_dir="./logs_test",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
    )

    print("\nStarting testing...")
    trainer = Trainer(
        model=model,
        args=test_args,
        eval_dataset=test_dataset,
        data_collator=ASRDataCollator(),
        compute_metrics=compute_metrics
    )

    print("Evaluating model on test set...")
    metrics = trainer.evaluate()

    print("\nGenerating predictions...")
    predictions = trainer.predict(test_dataset)

    if isinstance(predictions.predictions, tuple):
        logits = predictions.predictions[0]
    else:
        logits = predictions.predictions

    pred_ids = np.argmax(logits, axis=-1)

    mask = predictions.label_ids != -100

    pred_ids = pred_ids[mask]
    with open("metrics.txt", "w") as f:
        f.write("Speech-to-Text Model Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print("\nMetrics saved to 'metrics.txt'")


if __name__ == "__main__":
    test_and_visualize()