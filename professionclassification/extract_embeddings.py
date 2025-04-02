import argparse
import os
from pathlib import Path

import chromadb
import numpy as np
import torch
import wespeaker


def get_audio_path(audio_dir):

    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob('**/*.wav')) 
    print(f"Найдено {len(audio_files)} файлов в {audio_dir}")  # Добавляем отладочную печать
    return audio_files


def extract_embeddings(audio_files, device, pretraindir):  # Исправлено название аргумента
    model = wespeaker.load_model_local(pretraindir)
    model.set_device(device)

    embeddings = []

    for file_path in audio_files:
        try:
            embedding = model.extract_embedding(file_path)
            embedding = embedding.cpu().numpy()
            embeddings.append({
                'file_path': str(file_path),
                'embedding': embedding
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    return embeddings


def assign_labels(embeddings):
    folder_to_label = {
        "cooker": "повар",
        "programmer": "программист"
    }
    
    for emb in embeddings:
        class_name = Path(emb['file_path']).parent.name.lower()
        
        emb['label'] = folder_to_label.get(class_name, "другое")
            


def save_to_npy(embeddings, save_dir):
    numpy_embs = np.array(embeddings)
    np.save(os.path.join(save_dir, "numpy_embs.npy"), numpy_embs)

def save_to_chromadb(embeddings, db_path, split):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="profession_embeddings") # Исправлено имя коллекции

    collection.add(
        ids=[f"{split}_{i}" for i in range(len(embeddings))],
        embeddings=[item['embedding'] for item in embeddings],
        metadatas=[{
            "file_path": item['file_path'], "label": item['label'],
            "split": split
        }
            for item in embeddings]
    )


def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--train_dir", type=str, default="./profession_classificator_convert/train_audio",
                         help="Path to train audio files.")
     parser.add_argument("--test_dir", type=str, default="./profession_classificator_convert/test_audio",
                         help="Path to test audio files.")
     parser.add_argument("--pretraindir", type=str, default="./pretraindir",
                         help="Path to wespeaker model pretrain_dir.")
     parser.add_argument("--output", type=str, required=True,
                          choices=["npy", "chromadb"],
                          help="Embeddings saving format: npy or chromadb.")
     parser.add_argument("--save_path", type=str, default="./embs",
                         help="Save path for calculated embeddings")
     args = parser.parse_args()

     if not os.path.exists(args.train_dir):
         raise FileNotFoundError(f"Folder {args.train_dir} does not exists.")
     if not os.path.exists(args.test_dir):
         raise FileNotFoundError(f"Folder {args.test_dir} does not exists.")

     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     train_audio_files = get_audio_path(args.train_dir)
     test_audio_files = get_audio_path(args.test_dir)

     # 1. Разделяем данные на train и test (если их нет в разных папках)
     all_audio_files = train_audio_files + test_audio_files
     np.random.shuffle(all_audio_files)
     train_size = int(0.8 * len(all_audio_files))  # Например, 80% для обучения
     train_files = all_audio_files[:train_size]
     test_files = all_audio_files[train_size:]

     train_embeddings = extract_embeddings(train_files, device, args.pretraindir)
     test_embeddings = extract_embeddings(test_files, device, args.pretraindir)

     assign_labels(train_embeddings)
     assign_labels(test_embeddings)

     # Add 'split' key to each embedding
     for emb in train_embeddings:
         emb['split'] = 'train'
     for emb in test_embeddings:
         emb['split'] = 'test'

     if args.output == "npy":
         os.makedirs(args.save_path, exist_ok=True)
         save_to_npy(train_embeddings + test_embeddings, args.save_path)
        # save_to_npy(test_embeddings, args.save_path) # save all to one npy
     elif args.output == "chromadb":
         save_to_chromadb(train_embeddings, args.save_path, split="train")
         save_to_chromadb(test_embeddings, args.save_path, split="test")


if __name__ == '__main__':
    main()