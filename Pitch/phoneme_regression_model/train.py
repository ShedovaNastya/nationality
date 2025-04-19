import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import wandb
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

wandb.login()
wandb.init(project="phoneme_pitch_regression", name="phoneme_pitch_model")

class PhonemeEmbedder(nn.Module):
    def __init__(self, phoneme_vocab_size, embedding_dim=128, hidden_dim=256, nhead=4, num_layers=3):
        super(PhonemeEmbedder, self).__init__()
        self.embedding = nn.Embedding(phoneme_vocab_size, embedding_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, phoneme_seq):
        # phoneme_seq: (batch, seq_len)
        embedded = self.embedding(phoneme_seq) * np.sqrt(self.embedding.embedding_dim)
        embedded = embedded + self.pos_encoder[:, :embedded.size(1), :]
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch, embedding_dim)
        transformer_out = self.transformer_encoder(embedded)
        transformer_out = transformer_out.permute(1, 0, 2)  # (batch, seq_len, embedding_dim)
        output = self.fc(transformer_out)  # (batch, seq_len, hidden_dim)
        return output


class PitchRegressionModel(nn.Module):
    def __init__(self, speech_embed_dim=256, phoneme_embed_dim=256, hidden_dim=512):
        super(PitchRegressionModel, self).__init__()
        self.fusion_layer = nn.Linear(speech_embed_dim + phoneme_embed_dim, hidden_dim)

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, speech_embeddings, phoneme_embeddings):
        fused = torch.cat(
            [speech_embeddings.unsqueeze(1).expand(-1, phoneme_embeddings.size(1), -1), phoneme_embeddings], dim=2)
        fused_features = self.fusion_layer(fused)

        batch_size, seq_len, _ = fused_features.shape
        fused_flat = fused_features.view(batch_size * seq_len, -1)  # (batch*seq_len, hidden_dim)
        pitch_pred = self.regression_head(fused_flat)
        pitch_pred = pitch_pred.view(batch_size, seq_len)  # (batch, seq_len)

        return pitch_pred

class PhonemePitchDataset(Dataset):
    def __init__(self,
                 embeddings_path,
                 labels_path,
                 max_phoneme_length=90):
        # Загрузка речевых эмбеддингов
        with open(embeddings_path, 'rb') as f:
            embeddings_data = np.load(f, allow_pickle=True)
            all_embeddings = embeddings_data[0]['train'] + embeddings_data[0]['test']
            self.speech_embeddings = all_embeddings

        # Создаем словарь для быстрого поиска эмбеддингов
        self.embedding_dict = {}
        for emb in self.speech_embeddings:
            filename = os.path.splitext(os.path.basename(str(emb['file_path'])))[0]
            self.embedding_dict[filename] = emb

        # Сначала задаём max_phoneme_length, чтобы он был доступен в методах
        self.max_phoneme_length = max_phoneme_length

        # Загрузка и препроцессинг меток
        self.labels_df = pd.read_csv(labels_path, encoding='utf-8')
        self.labels_df = self.preprocess_pitch_data(self.labels_df)

        # Создание словаря фонем из всех последовательностей
        self.phoneme_to_idx = self.create_phoneme_vocab()

    def create_phoneme_vocab(self):
        all_phonemes = []
        for seq in self.labels_df['Фонемы']:
            all_phonemes.extend(seq.split())
        unique_phonemes = sorted(set(all_phonemes))
        return {phoneme: idx + 1 for idx, phoneme in enumerate(unique_phonemes)}

    def preprocess_pitch_data(self, df):
        all_pitch_values = []

        def collect_pitch_values(pitch_value):
            if isinstance(pitch_value, str):
                for token in pitch_value.split():
                    token = token.strip().lower()
                    if token == 'n/a': continue
                    try:
                        all_pitch_values.append(float(token))
                    except:
                        pass
            elif isinstance(pitch_value, (float, int)):
                if not np.isnan(pitch_value):
                    all_pitch_values.append(float(pitch_value))

        # Собираем все валидные значения
        df['Средние питчи (Hz)'].apply(collect_pitch_values)

        # Рассчитываем среднее значение pitch
        self.mean_pitch = np.mean(all_pitch_values) if all_pitch_values else 0.0

        # Теперь обрабатываем с заменой N/A
        df['processed_pitch'] = df['Средние питчи (Hz)'].apply(self.pitch_to_list)
        return df

    def pitch_to_list(self, pitch_value):
        pitch_values = []

        if isinstance(pitch_value, float):
            if np.isnan(pitch_value):
                return [self.mean_pitch] * self.max_phoneme_length
            else:
                return [pitch_value] * self.max_phoneme_length

        if isinstance(pitch_value, str):
            for token in pitch_value.split():
                token = token.strip().lower()
                if token == 'n/a':
                    pitch_values.append(self.mean_pitch)
                else:
                    try:
                        pitch_values.append(float(token))
                    except:
                        pitch_values.append(self.mean_pitch)

        # Обрезка/добавление паддинга
        if len(pitch_values) > self.max_phoneme_length:
            return pitch_values[:self.max_phoneme_length]
        else:
            return pitch_values + [self.mean_pitch] * (self.max_phoneme_length - len(pitch_values))


    def phoneme_to_tensor(self, phoneme_str):
        # Разбиваем строку на отдельные фонемы
        phonemes = phoneme_str.split()
        phoneme_indices = [self.phoneme_to_idx.get(p, 0) for p in phonemes]
        if len(phoneme_indices) > self.max_phoneme_length:
            phoneme_indices = phoneme_indices[:self.max_phoneme_length]
        else:
            phoneme_indices += [0] * (self.max_phoneme_length - len(phoneme_indices))
        return torch.tensor(phoneme_indices, dtype=torch.long)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        filename = os.path.splitext(row['Файл'])[0]

        if filename not in self.embedding_dict:
            raise ValueError(f"Не найдено вложение для файла: {row['Файл']}")

        speech_embedding = self.embedding_dict[filename]['embedding']
        phoneme_tensor = self.phoneme_to_tensor(row['Фонемы'])
        # Преобразуем список питчей в тензор
        pitch_tensor = torch.tensor(row['processed_pitch'], dtype=torch.float32)

        return {
            'speech_embeddings': torch.tensor(speech_embedding, dtype=torch.float32),  # (speech_embed_dim)
            'phoneme_seq': phoneme_tensor,  # (max_phoneme_length)
            'pitch_target': pitch_tensor   # (max_phoneme_length)
        }

def train_phoneme_pitch_regression(
        embeddings_path='',
        labels_path='',
        batch_size=32,
        num_epochs=20,
        learning_rate=3e-4,
        validation_split=0.2
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_dataset = PhonemePitchDataset(embeddings_path, labels_path)

    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    phoneme_embedder = PhonemeEmbedder(
        phoneme_vocab_size=len(full_dataset.phoneme_to_idx) + 1
    ).to(device)
    pitch_regression_model = PitchRegressionModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(phoneme_embedder.parameters()) +
        list(pitch_regression_model.parameters()),
        lr=learning_rate,
        weight_decay=1e-5 #L2
    )

    train_losses = []
    val_losses = []
    train_mae_scores = []
    val_mae_scores = []
    train_r2_scores = []
    val_r2_scores = []

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        phoneme_embedder.train()
        pitch_regression_model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_targets = []

        for batch in train_loader:
            # Move to device
            speech_embeddings = batch['speech_embeddings'].to(device)
            phoneme_seq = batch['phoneme_seq'].to(device)
            pitch_targets = batch['pitch_target'].to(device)

            # Маскирование паддинга
            mask = (phoneme_seq != 0).float().to(device)  # (batch, seq_len)

            # Phoneme embedding
            phoneme_embeds = phoneme_embedder(phoneme_seq)

            # Pitch prediction
            pitch_pred = pitch_regression_model(speech_embeddings, phoneme_embeds)

            # Compute loss with mask
            loss = torch.sum((pitch_pred - pitch_targets) ** 2 * mask) / (torch.sum(mask) + 1e-8)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(phoneme_embedder.parameters()) + list(pitch_regression_model.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            total_train_loss += loss.item()

            # Collect predictions for metrics
            all_train_preds.extend(pitch_pred.squeeze().detach().cpu().numpy())
            all_train_targets.extend(pitch_targets.detach().cpu().numpy())

        # Validation phase
        phoneme_embedder.eval()
        pitch_regression_model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                speech_embeddings = batch['speech_embeddings'].to(device)
                phoneme_seq = batch['phoneme_seq'].to(device)
                pitch_targets = batch['pitch_target'].to(device)

                # Phoneme embedding
                phoneme_embeds = phoneme_embedder(phoneme_seq)

                # Pitch prediction
                pitch_pred = pitch_regression_model(speech_embeddings, phoneme_embeds)

                mask = (phoneme_seq != 0).float().to(device)

                # Compute loss
                val_loss = torch.sum((pitch_pred - pitch_targets) ** 2 * mask) / (torch.sum(mask) + 1e-8)
                total_val_loss += val_loss.item()

                # Collect predictions for metrics
                all_val_preds.extend(pitch_pred.squeeze().detach().cpu().numpy())
                all_val_targets.extend(pitch_targets.detach().cpu().numpy())

        # Compute metrics
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)

        train_mae = mean_absolute_error(all_train_targets, all_train_preds)
        val_mae = mean_absolute_error(all_val_targets, all_val_preds)

        train_r2 = r2_score(all_train_targets, all_train_preds)
        val_r2 = r2_score(all_val_targets, all_val_preds)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mae_scores.append(train_mae)
        val_mae_scores.append(val_mae)
        train_r2_scores.append(train_r2)
        val_r2_scores.append(val_r2)

        # Logging
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")
        print(f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_r2": train_r2,
            "val_r2": val_r2
        })

    # Visualize training metrics
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # MAE plot
    plt.subplot(1, 3, 2)
    plt.plot(train_mae_scores, label='Train MAE')
    plt.plot(val_mae_scores, label='Validation MAE')
    plt.title('Mean Absolute Error over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # R2 Score plot
    plt.subplot(1, 3, 3)
    plt.plot(train_r2_scores, label='Train R2')
    plt.plot(val_r2_scores, label='Validation R2')
    plt.title('R2 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    wandb.log({"metrics_plot": wandb.Image('training_metrics.png')})

    # Преобразуем списки в numpy массивы
    all_val_targets = np.concatenate(all_val_targets).flatten()
    all_val_preds = np.concatenate(all_val_preds).flatten()

    # Scatter plot of predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(all_val_targets, all_val_preds, alpha=0.5, label='Predictions')
    plt.plot([min(all_val_targets), max(all_val_targets)],
             [min(all_val_targets), max(all_val_targets)],
             color='red', linestyle='--', label='Ideal Prediction')

    plt.title('Predictions vs Actual Pitch Values')
    plt.xlabel('Actual Pitch')
    plt.ylabel('Predicted Pitch')
    plt.legend()
    plt.grid(True)  # Добавить сетку для читаемости
    plt.savefig('predictions_scatter.png')
    wandb.log({"predictions_scatter": wandb.Image('predictions_scatter.png')})

    # Save models
    torch.save(phoneme_embedder.state_dict(), 'phoneme_embedder.pth')
    torch.save(pitch_regression_model.state_dict(), 'pitch_regression_model.pth')

    # Print final metrics
    print("\nFinal Training Metrics:")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Train MAE: {train_mae_scores[-1]:.4f}")
    print(f"Final Validation MAE: {val_mae_scores[-1]:.4f}")
    print(f"Final Train R2: {train_r2_scores[-1]:.4f}")
    print(f"Final Validation R2: {val_r2_scores[-1]:.4f}")


if __name__ == "__main__":
    train_phoneme_pitch_regression()
