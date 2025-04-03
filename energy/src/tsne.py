import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import seaborn as sns

from energy.src.train import PhonemeDataset

NUM_PHONEMES = 10
SAMPLES_PER_PHONEME = 10
SAVE_PATH = "../data/result/tsne_plot2.png"


def tsne():
    dataset = PhonemeDataset(csv_path="../data/phoneme_energy_eval.csv", pickle_path="embeds.pkl", is_train=False)

    phoneme_samples = {}

    for i in tqdm(range(len(dataset.data)), desc="phoneme collect"):
        item = dataset.__getitem__(i, row_label=True)

        label = item["labels"]
        embedding = np.hstack([item["energy"].numpy(), item["speech_embeddings"].numpy(), ])

        if label not in phoneme_samples and len(phoneme_samples) < NUM_PHONEMES:
            phoneme_samples[label] = [embedding]
        elif label in phoneme_samples and len(phoneme_samples[label]) < SAMPLES_PER_PHONEME:
            phoneme_samples[label].append(embedding)

    filtered_embeddings = []
    filtered_labels = []

    for phoneme in tqdm(phoneme_samples, desc="phoneme go through"):
        examples = phoneme_samples[phoneme]

        filtered_embeddings.extend(examples)

        filtered_labels.extend([phoneme] * len(examples))

    if not filtered_embeddings:
        print("Ошибка: нет данных для t-SNE!")
        return

    X = np.vstack(filtered_embeddings)
    y = np.array(filtered_labels)

    tsne = TSNE(n_components=2, perplexity=min(len(X) // 10, 50), random_state=42, max_iter=1500)
    X_tsne = tsne.fit_transform(X)

    unique_labels = list(set(y))
    palette = sns.color_palette("hsv", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in y]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='jet', alpha=0.7)

    for i, phoneme in enumerate(y):
        plt.annotate(phoneme, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8, alpha=0.7)

    print(f"Среднее эмбеддингов: {np.mean(X, axis=0)}")
    print(f"STD эмбеддингов: {np.std(X, axis=0)}")

    plt.title("t-SNE визуализация фонем")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.savefig("tsne_plot.png", dpi=300)
    print(f"График сохранен в tsne_plot.png")


if __name__ == '__main__':
    tsne()
