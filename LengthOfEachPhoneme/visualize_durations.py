import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pickle
import chromadb
from LengthOfEachPhoneme.train import Speech2TextModel, SpeechDataset
from transformers import T5Tokenizer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.manifold import TSNE
import random

def load_model_and_data(model_path, dataset_path, db_path, collection_name):
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загружаем модель с правильным количеством фонем (69)
    model = Speech2TextModel(t5_model_name="t5-base", num_phonemes=69)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Загружаем датасет
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    dataset = SpeechDataset(dataset_path, db_path, collection_name, tokenizer, train=False)
    
    return model, dataset

def visualize_duration_distribution(dataset, save_path):
    """Визуализация распределения длительностей фонем"""
    all_durations = []
    duration_labels = []
    
    for item in dataset:
        # Получаем длительности из датасета
        durations = item['durations'].numpy()  # Используем предварительно вычисленные длительности
        labels = item['duration_labels']
        
        all_durations.extend(durations)
        duration_labels.extend(labels)
    
    # Создаем график
    plt.figure(figsize=(12, 6))
    
    # Распределение длительностей
    plt.subplot(1, 2, 1)
    sns.histplot(data=all_durations, bins=50)
    plt.title('Distribution of Phoneme Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    
    # Распределение меток длительности
    plt.subplot(1, 2, 2)
    sns.countplot(x=duration_labels)
    plt.title('Distribution of Duration Labels')
    plt.xlabel('Duration Label')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path / 'duration_distribution.png')
    plt.close()

def save_metrics(all_labels, all_predictions, save_path):
    """Сохранение метрик в текстовый файл"""
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    with open(save_path / 'metrics.txt', 'w') as f:
        f.write("Результаты обучения модели:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write("\nМатрица ошибок:\n")
        f.write("-" * 30 + "\n")
        cm = confusion_matrix(all_labels, all_predictions)
        f.write("           short  medium  long\n")
        f.write("short     {:6d}  {:6d}  {:6d}\n".format(cm[0,0], cm[0,1], cm[0,2]))
        f.write("medium    {:6d}  {:6d}  {:6d}\n".format(cm[1,0], cm[1,1], cm[1,2]))
        f.write("long      {:6d}  {:6d}  {:6d}\n".format(cm[2,0], cm[2,1], cm[2,2]))

def visualize_model_predictions(model, dataset, save_path):
    """Визуализация предсказаний модели"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for item in dataset:
            speech_emb = item['speech_embeddings'].to(device).unsqueeze(0)
            labels = item['labels']
            durations = item['durations'].to(device).unsqueeze(0)
            phoneme_ids = item['phoneme_ids'].to(device).unsqueeze(0)
            
            # Получаем предсказания
            outputs = model(
                speech_embeddings=speech_emb,
                labels=labels.unsqueeze(0),
                phoneme_ids=phoneme_ids,
                durations=durations
            )
            predictions = torch.argmax(outputs["logits"], dim=-1)
            
            # Сохраняем предсказания и метки
            all_predictions.extend(predictions[0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Создаем матрицу ошибок
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(all_labels, all_predictions), 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=['short', 'medium', 'long'],
                yticklabels=['short', 'medium', 'long'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path / 'confusion_matrix.png')
    plt.close()
    
    # Сохраняем метрики
    save_metrics(all_labels, all_predictions, save_path)

def visualize_embeddings_tsne(model, dataset, save_path, n_samples=200):
    """Визуализация эмбеддингов с помощью t-SNE"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Собираем эмбеддинги и метки
    embeddings = []
    labels = []
    
    # Выбираем случайные примеры
    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)
    selected_indices = dataset_indices[:n_samples]
    
    with torch.no_grad():
        for idx in selected_indices:
            item = dataset[idx]
            speech_emb = item['speech_embeddings'].to(device).unsqueeze(0)
            duration_label = item['duration_labels'][0]  # Берем первую метку из последовательности
            
            # Получаем эмбеддинги из энкодера
            encoder_embeds = model.speech_encoder(speech_emb)
            embeddings.append(encoder_embeds[0].cpu().numpy().flatten())  # Берем среднее по временной оси
            labels.append(duration_label)
    
    # Преобразуем списки в массивы
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Применяем t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Создаем визуализацию
    plt.figure(figsize=(10, 8))
    
    # Создаем scatter plot для каждого класса отдельно
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Более различимые цвета
    labels_set = sorted(set(labels))
    
    for i, label in enumerate(labels_set):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=colors[i], label=['short', 'medium', 'long'][i],
                   alpha=0.6)
    
    plt.title('t-SNE visualization of phoneme embeddings')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'embeddings_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Пути к файлам
    model_path = "length_of_each_phoneme_en_model/checkpoint-600/pytorch_model.bin"
    dataset_path = "dataset/phoneme_dataset.pkl"
    db_path = "embeddings/"
    collection_name = "gender_embeddings"
    save_path = Path("visualization_results")
    save_path.mkdir(exist_ok=True)
    
    # Загружаем модель и данные
    model, dataset = load_model_and_data(model_path, dataset_path, db_path, collection_name)
    
    # Создаем визуализации
    visualize_duration_distribution(dataset, save_path)
    visualize_model_predictions(model, dataset, save_path)
    visualize_embeddings_tsne(model, dataset, save_path)
    
    print(f"Визуализации сохранены в директории: {save_path}")

if __name__ == "__main__":
    main() 