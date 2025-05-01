import argparse
import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch
from tqdm import tqdm
import wespeaker

from cca_similarity.cca_analysis import GetActivations, get_activations
from cca_similarity.cca_score import CCA


def load_phoneme_dataset(dataset_path):
    """
    Загружает датасет с фонемами и их длительностями.
    """
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_audio_files_from_dataset(data, subset='train', max_samples=100):
    """
    Получает пути к аудиофайлам из датасета фонем.
    """
    all_files = [item['audio_path'] for item in data[subset]]
    
    # Ограничиваем количество файлов для анализа
    if max_samples and len(all_files) > max_samples:
        import random
        random.seed(42)  # Для воспроизводимости
        all_files = random.sample(all_files, max_samples)
    
    return all_files


def get_duration_labels(audio_path, data, subset='train'):
    """
    Получает метки длительности для аудиофайла.
    """
    for item in data[subset]:
        if item['audio_path'] == audio_path:
            duration_labels = item['duration_labels']
            # Считаем частоту каждой метки
            counts = {'short': 0, 'medium': 0, 'long': 0}
            for dur in duration_labels:
                counts[dur] += 1
            
            # Определяем преобладающую метку
            predominant_label = max(counts, key=counts.get)
            return predominant_label, counts
    
    raise ValueError(f"Аудиофайл {audio_path} не найден в датасете")


def create_phoneme_duration_encoder():
    """
    Создает OneHotEncoder для меток длительности фонем.
    """
    duration_categories = ['short', 'medium', 'long']
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.array(duration_categories).reshape(-1, 1))
    return encoder


def compute_phoneme_duration_cca(activations, one_hot_label):
    """
    Вычисляет CCA между активациями модели и меткой длительности.
    """
    cca_coefs = []
    layers = [list(act.keys())[0] for act in activations]
    
    for act in activations:
        for layer_name, act_tensor in act.items():
            # Преобразуем активации в формат для CCA
            act_np = act_tensor.cpu().view(act_tensor.size(0), -1).numpy()
            
            # Повторяем метки для соответствия размеру активаций
            labels_repeated = np.tile(
                one_hot_label.flatten(), (act_np.shape[0], 1)
            )
            
            # Проверяем размеры и корректируем при необходимости
            if labels_repeated.shape[1] > act_np.shape[1]:
                labels_repeated = labels_repeated[:, :act_np.shape[1]]
            else:
                repeat_times = act_np.shape[1] // labels_repeated.shape[1] + 1
                labels_repeated = np.tile(labels_repeated, (1, repeat_times))
                labels_repeated = labels_repeated[:, :act_np.shape[1]]
            
            # Вычисляем CCA
            cca = CCA(act_np, labels_repeated)
            cca_results = cca.get_cca_parameters(
                epsilon_x=1e-6,
                epsilon_y=1e-6,
                verbose=False
            )
            
            cca_coefs.append(np.mean(cca_results["cca_coef1"]))
    
    return {'layers': layers, 'cca_coefs': cca_coefs}


def visualize_phoneme_duration_cca(duration_cca_results, save_path):
    """
    Визуализирует результаты CCA для разных длительностей фонем.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Для каждой метки длительности рисуем линию на графике
    for duration, results in duration_cca_results.items():
        layers = results['layers']
        coefs = results['cca_coefs']
        plt.plot(range(len(coefs)), coefs, label=f'Duration: {duration}', 
                marker='o', linewidth=2)
    
    plt.title("CCA Similarity: Model Activations vs Phoneme Duration")
    plt.xlabel("Layer")
    plt.ylabel("CCA Similarity")
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_phoneme_duration_cca(duration_cca_results, save_path):
    """
    Сохраняет результаты CCA в текстовый файл.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("CCA Similarity: Model Activations vs Phoneme Duration\n")
        f.write("=" * 60 + "\n\n")
        
        for duration, results in duration_cca_results.items():
            f.write(f"Duration: {duration}\n")
            f.write("-" * 40 + "\n")
            
            layers = results['layers']
            coefs = results['cca_coefs']
            
            for i, (layer, coef) in enumerate(zip(layers, coefs)):
                f.write(f"{layer}: {coef:.6f}\n")
            
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Calculate CCA for phoneme duration")
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        required=True,
        help="Path to wespeaker model pretrain_dir"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/phoneme_dataset.pkl",
        help="Path to phoneme dataset with duration labels"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of audio samples to analyze"
    )
    parser.add_argument(
        "--visual_save_path",
        type=str,
        default="./result/phoneme_duration_cca.png",
        help="Save path for visualization result"
    )
    parser.add_argument(
        "--text_save_path",
        type=str,
        default="./result/phoneme_duration_cca.txt",
        help="Save path for text result"
    )
    
    args = parser.parse_args()
    
    # Проверяем наличие директорий
    if not os.path.exists(args.pretrain_dir):
        raise FileNotFoundError(f"Folder {args.pretrain_dir} does not exist")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file {args.dataset_path} does not exist")
    
    # Создаем директорию для результатов
    os.makedirs(os.path.dirname(args.visual_save_path), exist_ok=True)
    
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Загружаем модель
    print(f"Loading model from {args.pretrain_dir}...")
    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)
    acts_model = GetActivations(model)
    
    # Загружаем датасет
    print(f"Loading dataset from {args.dataset_path}...")
    data = load_phoneme_dataset(args.dataset_path)
    
    # Получаем аудиофайлы из датасета
    audio_files = get_audio_files_from_dataset(data, max_samples=args.max_samples)
    print(f"Processing {len(audio_files)} audio files")
    
    # Создаем энкодер для меток длительности
    duration_encoder = create_phoneme_duration_encoder()
    
    # Словарь для хранения результатов CCA по длительностям
    duration_cca_results = {
        'short': {'cca_sum': None, 'count': 0, 'layers': None},
        'medium': {'cca_sum': None, 'count': 0, 'layers': None}, 
        'long': {'cca_sum': None, 'count': 0, 'layers': None}
    }
    
    # Обрабатываем аудиофайлы
    for audio_path in tqdm(audio_files, desc="Computing CCA"):
        try:
            # Получаем преобладающую метку длительности
            duration_label, _ = get_duration_labels(audio_path, data)
            
            # Преобразуем метку в one-hot вектор
            oh_label = duration_encoder.transform([[duration_label]])
            
            # Получаем активации модели
            acts = get_activations(acts_model, audio_path, device)
            
            # Вычисляем CCA
            cca_results = compute_phoneme_duration_cca(acts['act'], oh_label)
            
            # Обновляем результаты для соответствующей длительности
            if duration_cca_results[duration_label]['cca_sum'] is None:
                duration_cca_results[duration_label]['cca_sum'] = np.array(cca_results['cca_coefs'])
                duration_cca_results[duration_label]['layers'] = cca_results['layers']
            else:
                duration_cca_results[duration_label]['cca_sum'] += np.array(cca_results['cca_coefs'])
            
            duration_cca_results[duration_label]['count'] += 1
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    # Вычисляем средние значения CCA
    averaged_results = {}
    for duration, data in duration_cca_results.items():
        if data['count'] > 0:
            averaged_results[duration] = {
                'layers': data['layers'],
                'cca_coefs': data['cca_sum'] / data['count']
            }
    
    # Визуализируем результаты
    visualize_phoneme_duration_cca(averaged_results, args.visual_save_path)
    print(f"CCA visualization saved to {args.visual_save_path}")
    
    # Сохраняем результаты в текстовый файл
    save_phoneme_duration_cca(averaged_results, args.text_save_path)
    print(f"CCA results saved to {args.text_save_path}")


if __name__ == "__main__":
    main() 