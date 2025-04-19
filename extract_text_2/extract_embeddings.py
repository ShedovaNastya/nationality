import os
from pathlib import Path
import concurrent.futures
import chromadb
import numpy as np
import torch
import wespeaker
from sklearn.model_selection import train_test_split

DATA_DIR = "dataWW"
MODEL_DIR = "voxblink2_samresnet34"
SAVE_PATH = "embeddings"
OUTPUT_FORMAT = "chromadb"
BATCH_SIZE = 1000
TEST_SIZE = 0.2
RANDOM_SEED = 42


def load_model(model_dir):
    """Load WeSpeaker model"""
    model_path = os.path.join(model_dir, "avg_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file has not found: {model_path}")
    model = wespeaker.load_model_local(model_dir)
    return model


def get_audio_data(data_dir):
    """Collecting data from folders"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory has not found: {data_dir}")

    files = []
    for label_dir in Path(data_dir).iterdir():
        if label_dir.is_dir():
            files.extend([{
                "path": audio_file,
                "label": label_dir.name
            } for audio_file in label_dir.glob("*.wav")])

    print(f"Found {len(files)} audiofiles")
    return files


def process_file(file_info, model):
    """Process audio file"""
    try:
        embedding = model.extract_embedding(str(file_info["path"]))
        embedding = embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
        embedding = embedding.flatten()

        if embedding.shape != (256,):
            raise ValueError(f"Incorrect embedding size: {embedding.shape}")

        return {
            'file_path': str(file_info["path"]),
            'embedding': embedding,
            'label': file_info["label"]
        }
    except Exception as e:
        print(f"failed to process {file_info['path']}: {str(e)}")
        return None


def split_data(files):
    """split data for train/test"""
    train_files, test_files = train_test_split(
        files,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=[f["label"] for f in files]
    )
    print(f"\nsplit data: {len(train_files)} train, {len(test_files)} test")
    return train_files, test_files


def save_to_chromadb(results, collection_name, split_name):
    """Saving to ChromaDB with split indication"""
    client = chromadb.PersistentClient(path=SAVE_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    batch = []
    for i, item in enumerate(results):
        batch.append({
            "id": f"{split_name}_{i}",
            "embedding": item["embedding"].tolist(),
            "metadata": {
                "file_path": item["file_path"],
                "label": item["label"],
                "split": split_name
            }
        })

        if len(batch) >= BATCH_SIZE:
            collection.upsert(
                ids=[x["id"] for x in batch],
                embeddings=[x["embedding"] for x in batch],
                metadatas=[x["metadata"] for x in batch]
            )
            batch = []

    if batch:
        collection.upsert(
            ids=[x["id"] for x in batch],
            embeddings=[x["embedding"] for x in batch],
            metadatas=[x["metadata"] for x in batch]
        )


def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device: {device}")

    print("Load model...")
    try:
        model = load_model(MODEL_DIR)
        model.set_device(device)
    except Exception as e:
        print(f"failed to load model: {str(e)}")
        return

    audio_files = get_audio_data(DATA_DIR)
    train_files, test_files = split_data(audio_files)

    def process_files(files, split_name):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_file, file, model): file for file in files}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    result["split"] = split_name
                    results.append(result)
        print(f"Process {len(results)}/{len(files)} {split_name} files")
        return results

    train_results = process_files(train_files, "train")
    test_results = process_files(test_files, "test")

    print("\nsaving results...")
    if OUTPUT_FORMAT == "npy":
        np.save(os.path.join(SAVE_PATH, "train_embeddings.npy"), np.array([r["embedding"] for r in train_results]))
        np.save(os.path.join(SAVE_PATH, "test_embeddings.npy"), np.array([r["embedding"] for r in test_results]))
        print(f"Saved {len(train_results) + len(test_results)} embeddings")
    else:
        save_to_chromadb(train_results, "speech_embeddings", "train")
        save_to_chromadb(test_results, "speech_embeddings", "test")
        print("data saved into ChromaDB collection 'speech_embeddings'")


if __name__ == "__main__":
    main()
