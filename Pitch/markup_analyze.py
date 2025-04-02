import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv("markup_labels.csv", encoding='utf-8')

# Извлечение длин последовательностей фонем
df['phoneme_length'] = df['Фонемы'].apply(lambda x: len(x.split()))

# Статистика
print("Описательная статистика:")
print(df['phoneme_length'].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

# Визуализация
plt.figure(figsize=(10, 5))
plt.hist(df['phoneme_length'], bins=50, edgecolor='black')
plt.title("Распределение длин последовательностей фонем")
plt.xlabel("Длина")
plt.ylabel("Количество примеров")
plt.grid(True)
plt.show()