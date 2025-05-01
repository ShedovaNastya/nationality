import argparse
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import torch
import wespeaker


def get_audio_paths(metadata_path):
    metadata_path = Path(metadata_path).resolve()
    df = pd.read_csv(metadata_path)
    audio_files = [metadata_path.parent / path for path in df["audio_path"]]
    return audio_files, df["is_correct"].tolist()


def extract_embeddings(audio_files, device, pretrain_dir):
    model = wespeaker.load_model_local(pretrain_dir)
    model.set_device(device)

    embeddings = []

    for file_path in audio_files:
        embedding = model.extract_embedding(str(file_path))
        embedding = embedding.cpu().numpy()
        embeddings.append(
            {"file_path": str(file_path), "embedding": embedding.flatten()}
        )

    return embeddings

def assign_labels(embeddings, labels):
    for emb, label in zip(embeddings, labels):
        emb["label"] = int(label)
        emb["is_correct"] = bool(label)


def save_to_chromadb(embeddings, db_path):
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name="order_embeddings")

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(embeddings))
    shuffled_embeddings = [embeddings[i] for i in shuffled_indices]
    split_point = int(len(shuffled_embeddings) * 0.8)
    train_embeddings = shuffled_embeddings[:split_point]
    test_embeddings = shuffled_embeddings[split_point:]

    for emb in train_embeddings:
        emb["split"] = "train"
    for emb in test_embeddings:
        emb["split"] = "test"

    collection.add(
        ids=[f"emb_{i}" for i in range(len(embeddings))],
        embeddings=[e["embedding"].tolist() for e in embeddings],
        metadatas=[
            {
                "file_path": e["file_path"],
                "label": e["label"],
                "is_correct": e["is_correct"],
                "split": e.get("split", "train"),
            }
            for e in embeddings
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio embeddings using WeSpeaker"
    )
    parser.add_argument(
        "--metadata", type=str, required=True, help="Path to metadata.csv"
    )
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        required=True,
        help="Path to wespeaker model pretrain_dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        choices=["npy", "chromadb"],
        help="Embeddings saving format",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./embeddings",
        help="Save path for calculated embeddings",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_files, labels = get_audio_paths(args.metadata)
    embeddings = extract_embeddings(audio_files, device, args.pretrain_dir)
    assign_labels(embeddings, labels)

    if args.output == "chromadb":
        save_to_chromadb(embeddings, args.save_path)


if __name__ == "__main__":
    main()
