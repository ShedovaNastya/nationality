import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import chromadb
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig
from transformers import TrainerCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ASRDataset(Dataset):
    def __init__(self, chroma_db_path="embeddings", split="train", max_length=128):
        self.split = split
        self.client = chromadb.PersistentClient(path=chroma_db_path)

        try:
            self.collection = self.client.get_collection("speech_embeddings")
        except ValueError as e:
            raise ValueError(f"Collection 'speech_embeddings' not found in {chroma_db_path}") from e

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
        self.max_length = max_length

        print(f"\nLoading {split} data from ChromaDB...")
        self.data = []

        results = self.collection.get(
            where={"split": split},
            include=["embeddings", "metadatas"]
        )

        if not results or "embeddings" not in results or "metadatas" not in results:
            raise ValueError(f"failed to load data for split={split}")

        for emb, meta in zip(results["embeddings"], results["metadatas"]):
            if not all(key in meta for key in ["label", "file_path"]):
                continue

            self.data.append({
                "embedding": np.array(emb),
                "label": meta["label"],
                "file_path": meta["file_path"]
            })

        if not self.data:
            raise ValueError(f"data does not exists for split={split} after filtration")

        print(f"Loaded {len(self.data)} {split} samples")

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

    def __init__(self, t5_model_name="t5-small", audio_emb_dim=256, **kwargs):
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

class CustomSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            kwargs['model'].save_pretrained(
                args.output_dir,
                state_dict=kwargs['model'].state_dict(),
                safe_serialization=False
            )

chroma_db_path = "embeddings"
model_name = "t5-base"
max_length = 32
batch_size = 32

print("\nPreparing datasets...")
train_dataset = ASRDataset(chroma_db_path, split="train", max_length=max_length)
val_dataset = ASRDataset(chroma_db_path, split="test", max_length=max_length)

print("Initializing model...")
tokenizer = T5Tokenizer.from_pretrained(model_name)

config = ASRModelConfig(t5_model_name="t5-base", audio_emb_dim=256)
model = ASRModel(config).to(device)

training_args = TrainingArguments(
    output_dir="./asr_wespeaker_model",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=3e-4,
    num_train_epochs=20,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    save_safetensors=False,
    eval_strategy="steps"
)


print("\nStarting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=ASRDataCollator(),
    tokenizer=tokenizer
)

try:
    trainer.train()
except Exception as e:
    print(f"failed to train: {e}")
    trainer.save_model("./asr_wespeaker_interrupted")
    raise

trainer.save_model("./asr_wespeaker_final")
