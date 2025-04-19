import pickle
from uuid import uuid4

import pandas as pd
import torch
import torchaudio
import wespeaker
from tqdm import tqdm


def extract():
    train_csv = "../data/phoneme_energy_test.csv"
    eval_csv = "../data/phoneme_energy_eval.csv"
    pretrain_dir = "../models/voxblink2_samresnet34"
    pickle_path = "embeds.pkl"

    train_data = pd.read_csv(train_csv)
    eval_data = pd.read_csv(eval_csv)

    model = wespeaker.load_model_local(pretrain_dir)
    model.set_device("mps" if torch.backends.mps.is_available() else "cpu")
    sample_rate = 16000

    def extract_features(wav_path, start_time, end_time):
        pcm, sr = torchaudio.load(
            uri=wav_path,
            frame_offset=int(start_time * sample_rate),
            num_frames=int((end_time - start_time) * sample_rate),
        )
        embeddings = model.extract_embedding_from_pcm(pcm, sr)
        return embeddings.tolist()

    # Создаем словарь для хранения эмбеддингов
    embeddings_dict = {
        "train": {},
        "eval": {},
    }

    for idx, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Extracting embeddings & energy from test_ds"):
        wav_path = row['wav_path']
        start_time = row['start_time']
        end_time = row['end_time']
        energy = row['energy']

        key = row['id']

        embeddings_dict["train"][key] = {"embeddings": extract_features(wav_path, start_time, end_time), "energy": energy}

    for idx, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc="Extracting energy from eval_ds"):
        wav_path = row['wav_path']
        start_time = row['start_time']
        end_time = row['end_time']
        energy = row['energy']

        key = row['id']
        embeddings_dict["eval"][key] = {"embeddings": extract_features(wav_path, start_time, end_time), "energy": energy}

    with open(pickle_path, "wb") as f:
        pickle.dump(embeddings_dict, f)

    print(f"Эмбеддинги сохранены в {pickle_path}")


if __name__ == "__main__":
    extract()

