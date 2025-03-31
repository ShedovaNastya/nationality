import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from energy.src.train import PhonemeDataset

NUM_PHONEMES = 50
SAMPLES_PER_PHONEME = 15
SAVE_PATH = "../data/result/tsne_plot.png"

def tsne():
    dataset = PhonemeDataset(csv_path="../data/phoneme_energy-27000.csv", pretrain_dir="../models/voxblink2_samresnet34")

    phoneme_samples = {}
    dataset.data = dataset.data[:4000]
    for sample in tqdm(dataset, desc="phonem group"):
        phoneme_idx = sample["labels"].item()
        phoneme_name = list(dataset.phoneme_dict.keys())[phoneme_idx]

        if phoneme_name not in phoneme_samples:
            phoneme_samples[phoneme_name] = []

        phoneme_samples[phoneme_name].append(sample["speech_embeddings"].numpy())

    selected_phonemes = random.sample(list(phoneme_samples.keys()), min(NUM_PHONEMES, len(phoneme_samples)))
    filtered_embeddings = []
    filtered_labels = []

    for phoneme in tqdm(selected_phonemes, "phoneme go through"):
        examples = phoneme_samples[phoneme]
        selected_examples = random.sample(examples, min(SAMPLES_PER_PHONEME, len(examples)))

        filtered_embeddings.extend(selected_examples)
        filtered_labels.extend([phoneme] * len(selected_examples))

    X = np.vstack(filtered_embeddings)
    y = np.array(filtered_labels)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.arange(len(y)), cmap='jet', alpha=0.7)

    for i, phoneme in enumerate(y):
        plt.annotate(phoneme, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8, alpha=0.7)

    plt.title("t-SNE визуализация фонем")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"График сохранен в {SAVE_PATH}")


if __name__ == '__main__':
    tsne()