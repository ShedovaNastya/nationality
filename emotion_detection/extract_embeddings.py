import argparse
import os
import json
from pathlib import Path

import chromadb
import numpy as np
import torch
import wespeaker


def get_audio_path(audio_dir):
    """
    Recursively finds all audio files in the specified directory.
    """
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("**/*.wav"))

    return audio_files


def extract_embeddings(audio_files, device, pretrain_dir):
    """
    Extracts embeddings from audio files using the WeSpeaker model
    """
    model = wespeaker.load_model_local(pretrain_dir)
    model.set_device(device)

    embeddings = []

    for file_path in audio_files:
        embedding = model.extract_embedding(file_path)

        embedding = embedding.cpu().numpy()
        embeddings.append({
            "file_path": str(file_path),
            "embedding": embedding
        })

    return embeddings


def assign_labels(embeddings, data_dir, split):
    """
    Assigns emotion labels from .jsonl file
    """
    metadata_filename = f"{split}_metadata.jsonl"
    metadata_path = os.path.join(data_dir, metadata_filename)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"File {metadata_path} does not exists.")

    metadata = {}
    with open(metadata_path, "r") as file:
        for line in file:
            entry = json.loads(line)
            audio_path = entry["audio_path"].replace("\\", os.sep)
            file_name = os.path.basename(audio_path)
            emotion = entry["annotator_emo"]
            metadata[file_name] = emotion

    for emb in embeddings:
        current_file_path = emb["file_path"]
        current_file_name = os.path.basename(current_file_path)
        
        if current_file_name in metadata:
            emb["label"] = metadata[current_file_name]
        else:
            raise ValueError(f"File {current_file_name} not found in {metadata_path}")


def save_to_npy(embeddings, save_dir):
    """
    Saves embeddings in .npy format
    """
    numpy_embs = np.array(embeddings)
    np.save(os.path.join(save_dir, "numpy_embs.npy"), numpy_embs)


def save_to_chromadb(embeddings, db_path, split):
    """
    Stores embeddings in ChromaDB
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="emotion_embeddings")

    collection.add(
        ids=[f"{split}_{i}" for i in range(len(embeddings))],
        embeddings=[item["embedding"].flatten().tolist() for item in embeddings],
        metadatas=[{
            "file_path": item["file_path"], 
            "label": item["label"],
            "split": split
        }
            for item in embeddings]
    )


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=".dataset/train",
                        help="Path to train audio files.")
    parser.add_argument("--test_dir", type=str, default=".dataset/test",
                        help="Path to test audio files.")
    parser.add_argument("--pretrain_dir", type=str, default="./pretrain_dir",
                        help="Path to wespeaker model pretrain_dir.")
    parser.add_argument("--output", type=str, required=True,
                        choices=["npy", "chromadb"],
                        help="Embeddings saving format: npy or chromadb.")
    parser.add_argument("--save_path", type=str, default="./embeddings",
                        help="Save path for calculated embeddings")
    args = parser.parse_args()

    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Folder {args.train_dir} does not exists.")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Folder {args.test_dir} does not exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_audio_files = get_audio_path(os.path.join(args.train_dir, "wavs"))
    test_audio_files = get_audio_path(os.path.join(args.test_dir, "wavs"))

    train_embeddings = extract_embeddings(train_audio_files, device,
                                          args.pretrain_dir)
    test_embeddings = extract_embeddings(test_audio_files, device,
                                         args.pretrain_dir)

    assign_labels(train_embeddings, args.train_dir, "train")
    assign_labels(test_embeddings, args.test_dir, "test")

    if args.output == "npy":
        os.makedirs(args.save_path, exist_ok=True)
        embeddings = [{"train": train_embeddings, "test": test_embeddings}]
        save_to_npy(embeddings, args.save_path)
    elif args.output == "chromadb":
        save_to_chromadb(train_embeddings, args.save_path, split="train")
        save_to_chromadb(test_embeddings, args.save_path, split="test")


if __name__ == '__main__':
    main()
