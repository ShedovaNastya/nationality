import argparse
import pickle
import pandas as pd
import wandb
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5EncoderModel
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


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
    def __init__(
            self,
            csv_path,
            pickle_path,
            is_train=True,
    ):
        self.data = pd.read_csv(csv_path)
        self.phoneme_dict = {phoneme: idx for idx, phoneme in enumerate(sorted(set(self.data['phoneme'])))}

        with open(pickle_path, "rb") as f:
            collection = "train" if is_train else "eval"
            self.data_dict = pickle.load(f)[collection]
            print(f"path: {csv_path} collection: {collection}; head: {list(self.data_dict.keys())[:10]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        phoneme = row['phoneme']
        id = row['id'].__str__()

        item = self.data_dict[id]
        embeddings = item["embeddings"]
        energy = item["energy"]

        return {
            "labels": torch.tensor(self.phoneme_dict[phoneme], dtype=torch.long),
            "speech_embeddings": torch.tensor(embeddings, dtype=torch.float32),
            "energy": torch.tensor(energy, dtype=torch.float32),
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

        self.batch_norm = nn.BatchNorm1d(self.encoder.config.d_model)

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
        combined_input = torch.cat((speech_embeddings, energy), dim=1)  # Объединяем эмбеддинги и энергию
        speech_embeddings = self.speech_encoder(combined_input)  # Приводим к размерности T5

        if len(speech_embeddings.shape) == 2:
            speech_embeddings = speech_embeddings.unsqueeze(1)  # Добавляем размерность для seq_length

        encoded = self.encoder(inputs_embeds=speech_embeddings).last_hidden_state[:, 0, :]
        encoded = self.batch_norm(encoded)  # Применяем Batch Normalization
        output = self.fc(encoded)
        return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--csv_path',
        type=str,
        help='Path to csv dataset',
        required=True,
    )

    parser.add_argument(
        '--csv_val_path',
        type=str,
        help='Path to csv validation dataset',
        required=True,
    )

    parser.add_argument(
        '--pickle_path',
        type=str,
        help='Path to pickle file',
        required=True,
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
        '--sample-rate',
        type=int,
        help='Sample rate',
        default=16000,
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size',
        default=128,
    )

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    args = parser.parse_args()

    wandb.login(key=args.wandb_token)
    wandb.init(project="inrerp", name="t5_text_extraction")

    train_dataset = PhonemeDataset(
        csv_path=args.csv_path,
        pickle_path=args.pickle_path,
        is_train=True,
    )

    val_dataset = PhonemeDataset(
        csv_path=args.csv_val_path,
        pickle_path=args.pickle_path,
        is_train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = PhonemeRegressor(t5_model_name=args.model_name)
    model.to(device)

    criterion = nn.HuberLoss(delta=1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_metrics = {
        "loss": [],
        "mae": [],
        "rmse": [],
        "r2": [],
    }

    eval_metrics = {
        "loss": [],
        "mae": [],
        "rmse": [],
        "r2": [],
    }

    num_epochs = 100
    best_val_loss = float("inf")

    for epoch in tqdm(range(num_epochs), "epochs"):
        train_losses = []
        train_predictions = []
        train_true_labels = []

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            embeddings = batch["speech_embeddings"].to(device)
            energy = batch["energy"].to(device)
            targets = batch["labels"].to(device).float()

            outputs = model(energy, embeddings).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Ограничиваем градиенты
            optimizer.step()
            train_loss += loss.item()

            train_predictions.extend(outputs.detach().cpu().numpy())
            train_true_labels.extend(targets.detach().cpu().numpy())

        train_loss /= len(train_loader)

        model.eval()
        eval_loss = 0.0

        eval_losses = []
        eval_predictions = []
        eval_true_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                embeddings = batch["speech_embeddings"].to(device)
                energy = batch["energy"].to(device)
                targets = batch["labels"].to(device).float()

                outputs = model(energy, embeddings).squeeze(-1)

                loss = criterion(outputs, targets)
                eval_loss += loss.item()

                eval_predictions.extend(outputs.detach().cpu().numpy())
                eval_true_labels.extend(targets.detach().cpu().numpy())

        eval_loss /= len(eval_loader)
        scheduler.step(eval_loss)  # Адаптивное уменьшение LR

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        train_mae = mean_absolute_error(train_true_labels, train_predictions)
        train_r2 = r2_score(train_true_labels, train_predictions)
        train_rmse = root_mean_squared_error(train_true_labels, train_predictions, )

        train_metrics["loss"].append(train_loss)
        train_metrics["mae"].append(train_mae)
        train_metrics["rmse"].append(train_rmse)
        train_metrics["r2"].append(train_r2)

        eval_mae = mean_absolute_error(eval_true_labels, eval_predictions)
        eval_r2 = r2_score(eval_true_labels, eval_predictions)
        eval_rmse = root_mean_squared_error(eval_true_labels, eval_predictions)

        eval_metrics["loss"].append(eval_loss)
        eval_metrics["mae"].append(eval_mae)
        eval_metrics["rmse"].append(eval_rmse)
        eval_metrics["r2"].append(eval_r2)

        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), "best_model.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_metrics["loss"], label="Train Loss")
    plt.plot(range(1, num_epochs + 1), eval_metrics["loss"], label="Evaluation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_metrics["mae"], label="Train MAE")
    plt.plot(range(1, num_epochs + 1), eval_metrics["mae"], label="Evaluation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig("mae_plot.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_metrics["r2"], label="Train R²")
    plt.plot(range(1, num_epochs + 1), eval_metrics["r2"], label="Evaluation R²")
    plt.xlabel("Epoch")
    plt.ylabel("R² Score")
    plt.legend()
    plt.savefig("r2_plot.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_metrics["rmse"], label="Train RMSE")
    plt.plot(range(1, num_epochs + 1), eval_metrics["rmse"], label="Evaluation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("rmse_plot.png")


if __name__ == '__main__':
    main()
