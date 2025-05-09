import argparse
import os
import csv
from pathlib import Path
import parselmouth
import chromadb
import numpy as np
import torch
import wespeaker
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_formants_cache(formants_csv):
    formants_cache = {}
    with open(formants_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = Path(row['FileName']).name
            try:
                f1, f2, f3 = float(row['F1']), float(row['F2']), float(row['F3'])
                if filename not in formants_cache:
                    formants_cache[filename] = []
                formants_cache[filename].append((f1, f2, f3))
            except:
                continue
    return formants_cache

def calculate_formants(audio_path, formants_cache):
    filename = audio_path.name
    if filename not in formants_cache or not formants_cache[filename]:
        return None
    avg_formants = np.mean(formants_cache[filename], axis=0)
    return avg_formants.tolist()

def get_audio_path(audio_dir):
    audio_dir = Path(audio_dir)
    return list(audio_dir.glob('**/*.wav')) + list(audio_dir.glob('**/*.mp3'))

def load_audio(file_path):
    import torchaudio
    try:
        pcm, sample_rate = torchaudio.load(str(file_path))
        return file_path, pcm, sample_rate
    except:
        return file_path, None, None

def extract_embeddings(audio_files, device, pretrain_dir, formants_cache, batch_size=256):
    model = wespeaker.load_model_local(pretrain_dir)
    model.set_device(device)
    print(f"Model is on device: {device}")
    embeddings = []
    with ThreadPoolExecutor() as executor:
        audio_data = list(tqdm(executor.map(load_audio, audio_files), total=len(audio_files)))
    valid_audio = [(fp, pcm, sr) for fp, pcm, sr in audio_data if pcm is not None]
    print(f"Loaded {len(valid_audio)} valid audio files out of {len(audio_files)}")
    for i in tqdm(range(0, len(valid_audio), batch_size)):
        batch = valid_audio[i:i + batch_size]
        batch_paths = [fp for fp, _, _ in batch]
        try:
            with torch.no_grad():
                batch_embeddings = [model.extract_embedding(str(fp)) for fp in batch_paths]
                batch_embeddings = [emb.cpu().numpy().flatten() for emb in batch_embeddings]
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            continue
        for file_path, embedding in zip(batch_paths, batch_embeddings):
            formants = calculate_formants(file_path, formants_cache)
            if formants is None:
                continue
            embeddings.append({
                'file_path': str(file_path),
                'embedding': embedding,
                'f1': formants[0],
                'f2': formants[1],
                'f3': formants[2]
            })
    return embeddings

def save_to_chromadb(embeddings, db_path, split):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="formants_embeddings")
    with tqdm(total=len(embeddings)):
        collection.add(
            ids=[f"{split}_{i}" for i in range(len(embeddings))],
            embeddings=[item['embedding'] for item in embeddings],
            metadatas=[{
                "file_path": item['file_path'],
                "f1": item['f1'],
                "f2": item['f2'],
                "f3": item['f3'],
                "split": split
            } for item in embeddings]
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--formants_file", type=str, required=True)
    parser.add_argument("--pretrain_dir", type=str, default="./voxblink2_samresnet34")
    parser.add_argument("--save_path", type=str, default="./embeddings")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    formants_cache = load_formants_cache(args.formants_file)
    train_audio_files = get_audio_path(args.train_dir)
    train_embeddings = extract_embeddings(train_audio_files, device,
                                         args.pretrain_dir, formants_cache, args.batch_size)
    test_audio_files = get_audio_path(args.test_dir)
    test_embeddings = extract_embeddings(test_audio_files, device,
                                        args.pretrain_dir, formants_cache, args.batch_size)
    os.makedirs(args.save_path, exist_ok=True)
    save_to_chromadb(train_embeddings, args.save_path, "train")
    save_to_chromadb(test_embeddings, args.save_path, "test")

if __name__ == '__main__':
    main()