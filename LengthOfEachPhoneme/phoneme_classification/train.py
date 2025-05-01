import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, TrainerCallback
import pickle
import chromadb
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


wandb.login()
wandb.init(project="inrerp", name="t5_phoneme_duration_en")



class SpeechEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=768):
        super(SpeechEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class Speech2TextModel(nn.Module):
    def __init__(self, t5_model_name="t5-base", num_phonemes=100):
        super(Speech2TextModel, self).__init__()
        self.speech_encoder = SpeechEncoder(input_dim=256, hidden_dim=768)
        
        # Эмбеддинги
        self.position_embeddings = nn.Embedding(512, 768)
        self.phoneme_embeddings = nn.Embedding(num_phonemes, 768)
        
        # Проекция длительностей
        self.duration_projection = nn.Sequential(
            nn.Linear(1, 768),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Слой слияния
        self.fusion_layer = nn.Linear(768 * 4, 768)  # 4 because we now have duration features
        self.layer_norm = nn.LayerNorm(768)
        
        # Transformer encoder для лучшего моделирования последовательности
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Классификатор
        self.duration_classifier = nn.Linear(768, 3)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, speech_embeddings, labels=None, phoneme_ids=None, durations=None):
        batch_size = speech_embeddings.shape[0]
        sequence_length = labels.shape[1] if labels is not None else durations.shape[1]
        
        # Базовые эмбеддинги
        speech_embeds = self.speech_encoder(speech_embeddings)
        repeated_speech_embeds = speech_embeds.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # Позиционные эмбеддинги
        position_ids = torch.arange(sequence_length, device=speech_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        # Эмбеддинги фонем
        if phoneme_ids is None:
            phoneme_ids = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=speech_embeddings.device)
        phoneme_embeds = self.phoneme_embeddings(phoneme_ids)
        
        # Проекция длительностей
        if durations is not None:
            duration_embeds = self.duration_projection(durations.unsqueeze(-1))
        else:
            duration_embeds = torch.zeros_like(repeated_speech_embeds)
        
        # Объединяем все признаки
        combined_embeds = torch.cat([
            repeated_speech_embeds,
            position_embeds,
            phoneme_embeds,
            duration_embeds
        ], dim=-1)
        
        # Слияние и нормализация
        fused_embeds = self.fusion_layer(combined_embeds)
        fused_embeds = self.layer_norm(fused_embeds)
        fused_embeds = self.dropout(fused_embeds)
        
        # Transformer для моделирования последовательности
        transformer_output = self.transformer(fused_embeds)
        transformer_output = self.dropout(transformer_output)
        
        # Классификация
        logits = self.duration_classifier(transformer_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_logits = logits.view(-1, 3)
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}

class SpeechDataset(Dataset):
    def __init__(self, path_to_pkl, path_to_db, collection_name, tokenizer, train=True, max_target_length=128):
        with open(path_to_pkl, "rb") as f:
            self.data = pickle.load(f)
            
        if train:
            self.data = self.data['train']
        else:
            self.data = self.data['dev']
            
        self.duration_to_idx = {'short': 0, 'medium': 1, 'long': 2}
        
        # Создаем словарь для фонем
        all_phonemes = set()
        for item in self.data:
            all_phonemes.update(item['phonemes'])
        self.phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(sorted(all_phonemes))}
        
        # Вычисляем статистики длительностей для нормализации
        all_durations = []
        for item in self.data:
            durations = [end - start for end, start in zip(item['end_times'], item['start_times'])]
            all_durations.extend(durations)
        self.duration_mean = sum(all_durations) / len(all_durations)
        self.duration_std = (sum((d - self.duration_mean) ** 2 for d in all_durations) / len(all_durations)) ** 0.5
        
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        
        self.client = chromadb.PersistentClient(path=path_to_db)
        self.collection_name = collection_name
        self.coll = self.client.get_collection(self.collection_name)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        phonemes = item['phonemes']
        duration_labels = item['duration_labels']
        
        # Вычисляем и нормализуем длительности
        durations = [(end - start) for end, start in zip(item['end_times'], item['start_times'])]
        normalized_durations = [(d - self.duration_mean) / self.duration_std for d in durations]
        
        # Convert labels and phonemes to indices
        duration_indices = torch.tensor([self.duration_to_idx[label] for label in duration_labels])
        phoneme_indices = torch.tensor([self.phoneme_to_idx[phoneme] for phoneme in phonemes])
        
        # Convert dataset ID to database ID format
        db_id = f"train_{idx}" if idx < len(self.data) else f"dev_{idx - len(self.data)}"
        
        # Get speaker embedding
        result = self.coll.get(ids=[db_id], include=["embeddings"])
        embedding = result["embeddings"][0] if result["embeddings"] is not None else None
        
        return {
            "speech_embeddings": torch.Tensor(embedding),
            "labels": duration_indices,
            "phoneme_ids": phoneme_indices,
            "durations": torch.tensor(normalized_durations, dtype=torch.float),
            "phonemes": phonemes,
            "duration_labels": duration_labels,
            "audio_path": item['audio_path']
        }

class CustomSpeechDataCollator:
    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
    
    def __call__(self, features):
        speech_embeddings = torch.stack([f["speech_embeddings"] for f in features])
        
        # Pad sequences
        max_len = max(len(f["labels"]) for f in features)
        padded_labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        padded_phoneme_ids = torch.full((len(features), max_len), 0, dtype=torch.long)
        padded_durations = torch.zeros((len(features), max_len), dtype=torch.float)
        
        for i, f in enumerate(features):
            labels = f["labels"]
            phoneme_ids = f["phoneme_ids"]
            durations = f["durations"]
            length = len(labels)
            
            padded_labels[i, :length] = labels
            padded_phoneme_ids[i, :length] = phoneme_ids
            padded_durations[i, :length] = durations

        return {
            "speech_embeddings": speech_embeddings,
            "labels": padded_labels,
            "phoneme_ids": padded_phoneme_ids,
            "durations": padded_durations
        }

class InferenceCallback(TrainerCallback):
    def __init__(self, sample_embedding, original_durations, sample_length=None):
        self.sample_embedding = sample_embedding
        self.original_durations = original_durations.split()  # Разбиваем строку на список
        self.sample_length = len(self.original_durations) if sample_length is None else sample_length
        self.idx_to_duration = {0: 'short', 1: 'medium', 2: 'long'}

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 200 == 0:
            model = kwargs.get("model")
            device = args.device
            
            sample_emb = self.sample_embedding.to(device).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                # Передаем длину последовательности через labels
                dummy_labels = torch.zeros((1, self.sample_length), dtype=torch.long, device=device)
                outputs = model(sample_emb, labels=dummy_labels)
                # logits теперь имеют размерность [batch_size, seq_len, 3]
                predicted_durations = torch.argmax(outputs["logits"][0, :self.sample_length], dim=-1)
                predicted_labels = [self.idx_to_duration[idx.item()] for idx in predicted_durations]

            # Log original and predicted durations
            original_text = "Original durations: " + " ".join(self.original_durations)
            predicted_text = "Predicted durations: " + " ".join(predicted_labels)

            wandb.log({
                "original": wandb.Html(original_text),
                "predicted": wandb.Html(predicted_text)
            })

            model.train()
        return control
    
    
if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    t5_model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    path_to_pkl = "dataset/phoneme_dataset.pkl"
    path_to_db = "embeddings/"
    db_collection_name = "gender_embeddings"

    train_dataset = SpeechDataset(path_to_pkl,path_to_db,db_collection_name, tokenizer)
    val_dataset = SpeechDataset(path_to_pkl,path_to_db,db_collection_name, tokenizer,train=False)

    # Создаем модель с правильным количеством фонем
    num_phonemes = len(train_dataset.phoneme_to_idx)
    model = Speech2TextModel(t5_model_name=t5_model_name, num_phonemes=num_phonemes)

    print(f"Number of phonemes in dataset: {num_phonemes}")
    print(f"Total trainable parameters: {count_parameters(model)}")
    data_collator = CustomSpeechDataCollator(tokenizer, max_length=128)

    training_args = TrainingArguments(
        output_dir="./speech2text_en_model",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        save_total_limit=10,
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    sample_embedding = train_dataset[0]['speech_embeddings'].to('cuda:0')

    class MetricsCallback(TrainerCallback):
        def __init__(self):
            self.idx_to_duration = {0: 'short', 1: 'medium', 2: 'long'}
            
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            model = kwargs.get("model")
            eval_dataloader = kwargs.get("eval_dataloader")
            device = args.device
            
            all_preds = []
            all_labels = []
            
            model.eval()
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with torch.no_grad():
                    outputs = model(**batch)
                    logits = outputs["logits"]
                    
                    # Получаем предсказания
                    preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
                    labels = batch["labels"]  # [batch_size, seq_len]
                    
                    # Маскируем padding
                    mask = (labels != -100)
                    preds = preds[mask].cpu().numpy()
                    labels = labels[mask].cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            
            # Вычисляем метрики
            accuracy = accuracy_score(all_labels, all_preds)
            conf_matrix = confusion_matrix(all_labels, all_preds)
            class_report = classification_report(all_labels, all_preds, 
                                            target_names=['short', 'medium', 'long'],
                                            output_dict=True)
            
            # Логируем в wandb
            wandb.log({
                "eval_accuracy": accuracy,
                "eval_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=['short', 'medium', 'long']
                ),
                "eval_short_f1": class_report['short']['f1-score'],
                "eval_medium_f1": class_report['medium']['f1-score'],
                "eval_long_f1": class_report['long']['f1-score'],
            })
            
            # Добавляем метрики в вывод
            metrics["eval_accuracy"] = accuracy
            for cls in ['short', 'medium', 'long']:
                metrics[f"eval_{cls}_f1"] = class_report[cls]['f1-score']
            
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            InferenceCallback(
                sample_embedding=sample_embedding,
                original_durations=" ".join(train_dataset[0]['duration_labels'])
            ),
            MetricsCallback()
        ]
    )
    trainer.args.save_safetensors = False
    trainer.train()

