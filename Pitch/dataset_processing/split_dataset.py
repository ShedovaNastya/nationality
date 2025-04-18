import os
import shutil
import random

# Исходная папка с аудиофайлами (путь к датасету)
source_dir = "dataset/LibriTTS_R/test-clean"

# Папка, куда сохраняем разделённые данные
destination_dir = "dataset_split"
train_dir = os.path.join(destination_dir, "train")
test_dir = os.path.join(destination_dir, "test")

# Создаём папки, если их нет
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Список всех аудиофайлов
all_files = []

# Проходим по всем папкам внутри исходного каталога
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".wav"):  # Фильтруем только аудиофайлы
            all_files.append(os.path.join(root, file))

# Перемешиваем файлы для случайного разделения
random.shuffle(all_files)

# 80% файлов — в train, 20% — в test
split_idx = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
test_files = all_files[split_idx:]


# Функция для копирования файлов
def copy_files(file_list, target_dir):
    for src in file_list:
        filename = os.path.basename(src)  # Получаем только имя файла
        dst = os.path.join(target_dir, filename)  # Новый путь

        # Проверяем, существует ли файл перед копированием
        if not os.path.exists(src):
            print(f"Ошибка: Файл {src} не найден!")
            continue

        shutil.copyfile(src, dst)
        print(f"Файл {filename} скопирован в {target_dir}")


# Копируем файлы
print("\n Копируем тренировочные файлы...")
copy_files(train_files, train_dir)

print("\n Копируем тестовые файлы...")
copy_files(test_files, test_dir)

print("\n Разделение датасета завершено!")
print(f"📂 {len(train_files)} файлов в train/, {len(test_files)} файлов в test/")
