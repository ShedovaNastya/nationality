import argparse
import wandb
import pandas as pd
import parselmouth
import librosa
import torch

from transformers import T5Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration


class CustomSpeechDataCollator:
    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features):
        max_seq_len = max(f["speech_embeddings"].shape[0] for f in features)

        speech_embeddings = [
            torch.nn.functional.pad(f["speech_embeddings"], (0, 0, 0, max_seq_len - f["speech_embeddings"].shape[0]))
            for f in features
        ]
        speech_embeddings = torch.stack(speech_embeddings)

        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        batch = {
            "speech_embeddings": speech_embeddings,
            "labels": padded_labels,
        }
        return batch


class SpeechDataset(Dataset):
    def __init__(
            self,
            csv_path,
            tokenizer,
            sample_rate=16000,
            n_mfcc=13,
    ):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract_features(self, wav_path, start_time, end_time, max_frames=30):
        y, sr = librosa.load(wav_path, sr=self.sample_rate)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        snd = parselmouth.Sound(segment, sampling_frequency=sr)
        mfcc = snd.to_mfcc(number_of_coefficients=self.n_mfcc).to_array()

        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        if mfcc.shape[1] > max_frames:
            mfcc = mfcc[:, :max_frames]
        if mfcc.shape[1] < max_frames:
            pad = torch.zeros(mfcc.shape[0], max_frames - mfcc.shape[1])
            mfcc = torch.cat((mfcc, pad), dim=1)

        return mfcc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        phoneme = row['phoneme']
        wav_path = row['wav_path']
        start_time = row['start_time']
        end_time = row['end_time']

        features = self.extract_features(wav_path, start_time, end_time)
        phoneme_ids = self.tokenizer.encode(phoneme)
        result = {
            "labels": torch.tensor(phoneme_ids, dtype=torch.long),
            "speech_embeddings": features,
        }
        # print(f"__getitem__[{idx}] ->", result)  # Debug print
        return result


class SpeechEncoder(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=768):
        super(SpeechEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x.transpose(1, 2))
        x = self.activation(x)
        return x


class Speech2TextModel(nn.Module):
    def __init__(self, t5_model_name="t5-base"):
        super(Speech2TextModel, self).__init__()

        # input dim = 13 MFCC
        self.speech_encoder = SpeechEncoder(input_dim=14, hidden_dim=768)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    def forward(self, speech_embeddings, labels=None):
        encoder_embeds = self.speech_encoder(speech_embeddings)

        print(f"embeds shape: {encoder_embeds.shape}\n")

        attention_mask = torch.ones(encoder_embeds.shape[:2], dtype=torch.long, device=encoder_embeds.device)
        print(f"Attention mask shape: {attention_mask.shape}")

        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            labels=labels
        )
        return outputs


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
        '--model_name',
        type=str,
        help='Name of model',
        choices=['t5-small', 't5-base', 't5-large'],
        default='google-t5/t5-base',
    )

    parser.add_argument(
        '--wandb_token',
        type=str,
        help='Wandb token',
        required=True,
    )


    args = parser.parse_args()

    wandb.login(key=args.wandb_token)
    wandb.init(project="inrerp", name="t5_text_extraction")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    train_dataset = SpeechDataset(args.csv_path, tokenizer)

    val_dataset = SpeechDataset(args.csv_val_path, tokenizer)

    model = Speech2TextModel(t5_model_name=args.model_name)

    data_collator = CustomSpeechDataCollator(tokenizer, max_length=128)

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
        tokenizer=tokenizer,
    )

    trainer.args.save_safetensors = False
    trainer.train()


if __name__ == '__main__':
    main()
