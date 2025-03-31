import argparse
import torchaudio
import wandb
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import wespeaker
from torch.utils.data import Dataset
from torch import nn
from transformers import Trainer, TrainingArguments, T5EncoderModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score


class CustomDataCollator:
    def __call__(self, features):
        speech_embeddings = torch.stack([f["speech_embeddings"] for f in features])
        energies = torch.tensor([f["energy"] for f in features], dtype=torch.float32)
        labels = [f["labels"] for f in features]
        return {
            "speech_embeddings": speech_embeddings,
            "energy": energies,
            "labels": labels
        }


class PhonemeDataset(Dataset):
    def __init__(self, csv_path, pretrain_dir, sample_rate=16000):
        self.data = pd.read_csv(csv_path)
        self.phoneme_dict = {phoneme: idx for idx, phoneme in enumerate(sorted(set(self.data['phoneme'])))}
        self.sample_rate = sample_rate
        self.model = wespeaker.load_model_local(pretrain_dir)
        self.model.set_device("mps" if torch.backends.mps.is_available() else "cpu")

    def extract_features(self, wav_path, start_time, end_time):
        pcm, sr = torchaudio.load(
            uri=wav_path,
            frame_offset=int(start_time * self.sample_rate),
            num_frames=int((end_time - start_time) * self.sample_rate),
        )
        embeddings = self.model.extract_embedding_from_pcm(pcm, sr)
        return embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        phoneme = row['phoneme']
        energy = row['energy']
        wav_path = row['wav_path']
        start_time = row['start_time']
        end_time = row['end_time']
        embeddings = self.extract_features(wav_path, start_time, end_time)
        return {
            "labels": torch.tensor(self.phoneme_dict[phoneme], dtype=torch.long),
            "speech_embeddings": torch.tensor(embeddings, dtype=torch.float32),
            "energy": torch.tensor(energy, dtype=torch.float32)
        }


class PhonemeRegressor(nn.Module):
    def __init__(
            self,
            t5_model_name='t5-small',
            embedding_dim=256,
            hidden_dim=512,
            output_dim=1,
            dropout_prob=0.3,
    ):
        super(PhonemeRegressor, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(t5_model_name)
        self.speech_encoder = nn.Linear(embedding_dim + 1, self.encoder.config.d_model)  # +1 для энергии

        self.fc = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, energy, speech_embeddings, labels=None):
        energy = energy.unsqueeze(1)
        # print(f"energy shape {energy.shape}")
        # print(f"speech_embedding shape {speech_embeddings.shape}")
        combined_input = torch.cat((speech_embeddings, energy), dim=1)  # Объединяем эмбеддинги и энергию
        speech_embeddings = self.speech_encoder(combined_input)  # Приводим к размерности T5

        if len(speech_embeddings.shape) == 2:  # Проверяем, что размерность только (batch_size, embedding_dim)
            speech_embeddings = speech_embeddings.unsqueeze(1)  # Добавляем размерность для seq_length

        encoded = self.encoder(inputs_embeds=speech_embeddings).last_hidden_state[:, 0, :]
        output = self.fc(encoded)
        return output


def extract_embeddings_and_labels(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            speech_embeddings = batch['speech_embeddings'].to(device)
            energy = torch.tensor(batch['energy'], dtype=torch.float32).to(device)
            label = batch['labels'].to(device)

            print(f"Speech embeddings shape: {speech_embeddings.shape}")
            print(f"Energy shape: {energy.shape}")

            output = model(energy, speech_embeddings)
            embeddings.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels


history = {
    "rmse": [],
    "mae": [],
    "r2": [],
}


def compute_metrics(eval_pred):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    predictions, labels = eval_pred

    # Приводим к torch.Tensor и переносим на device
    predictions = torch.tensor(predictions, dtype=torch.float32).flatten().to(device)
    labels = torch.tensor(np.array(labels), dtype=torch.long).flatten().to(device)

    if len(labels) != len(predictions):
        print(f"Размеры labels: {len(labels)}, predictions: {len(predictions)}")
        min_len = min(len(labels), len(predictions))
        labels = labels[:min_len]
        predictions = predictions[:min_len]

    # Вычисляем метрики
    rmse = np.sqrt(mean_squared_error(labels.cpu().numpy(), predictions.cpu().numpy()))
    mae = mean_absolute_error(labels.cpu().numpy(), predictions.cpu().numpy())
    r2 = r2_score(labels.cpu().numpy(), predictions.cpu().numpy())

    history["rmse"].append(rmse)
    history["mae"].append(mae)
    history["r2"].append(r2)

    return {
        "rmse": torch.tensor(rmse, dtype=torch.float32).to(device),
        "mae": torch.tensor(mae, dtype=torch.float32).to(device),
        "r2": torch.tensor(r2, dtype=torch.float32).to(device)
    }


def save_metrics_plots(history, output_dir='../data/result/'):
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, values in history.items():
        plt.figure()
        plt.plot(values)
        plt.title(f'{metric_name} over epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.grid(True)

        file_path = os.path.join(output_dir, f"{metric_name}.png")
        plt.savefig(file_path)
        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--csv_path',
        type=str,
        help='Path to csv dataset',
        required=True,
        default='../data/phoneme_energy-27000.csv',
    )

    parser.add_argument(
        '--csv_val_path',
        type=str,
        help='Path to csv validation dataset',
        required=True,
        default='../data/eval_dataset.csv'
    )

    parser.add_argument(
        '--wandb_token',
        type=str,
        help='Wandb token',
        required=True,
    )

    parser.add_argument(
        '--pretrain_dir',
        type=str,
        help='Pretrain dir',
        required=True,
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='Name of model',
        choices=['t5-small', 't5-base', 't5-large'],
        default='google-t5/t5-base',
    )

    parser.add_argument(
        '--tsne_plot_path',
        type=str,
        help='Path to save tsne_plot figure',
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        help='Sample rate',
        default=16000,
    )

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    args = parser.parse_args()

    wandb.login(key=args.wandb_token)
    wandb.init(project="inrerp", name="t5_text_extraction")

    train_dataset = PhonemeDataset(
        csv_path=args.csv_path,
        sample_rate=args.sample_rate,
        pretrain_dir=args.pretrain_dir,
    )

    val_dataset = PhonemeDataset(
        csv_path=args.csv_val_path,
        sample_rate=args.sample_rate,
        pretrain_dir=args.pretrain_dir,
    )

    model = PhonemeRegressor(t5_model_name=args.model_name)
    model.to(device)

    data_collator = CustomDataCollator()

    training_args = TrainingArguments(
        output_dir="speech2text_en_model",
        num_train_epochs=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=10,
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.args.save_safetensors = False
    trainer.train()

    save_metrics_plots(history)


if __name__ == '__main__':
    main()
