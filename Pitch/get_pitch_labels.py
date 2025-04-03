import os
import parselmouth
import numpy as np
import textgrid
import csv


def get_average_pitch(pitch, tmin, tmax):
    """
    Вычисляет среднее значение питча в интервале [tmin, tmax].
    Игнорирует нулевые или отрицательные значения (неопределённый питч).
    """
    times = pitch.xs()
    pitch_values = pitch.selected_array['frequency']
    # Находим индексы, соответствующие временному интервалу
    indices = np.where((times >= tmin) & (times <= tmax))[0]
    if len(indices) == 0:
        return None
    values = pitch_values[indices]
    # Отбрасываем невалидные (ноль и ниже)
    values = values[values > 0]
    if len(values) == 0:
        return None
    return np.mean(values)


def process_utterance(wav_path, textgrid_path):
    """
    Для заданного аудиофайла и соответствующего файла TextGrid
    вычисляет средний питч для каждого интервала (фонемы).
    """
    # Загружаем аудио
    sound = parselmouth.Sound(wav_path)
    pitch = sound.to_pitch()  # Можно задать дополнительные параметры, если нужно

    # Загружаем TextGrid
    tg = textgrid.TextGrid.fromFile(textgrid_path)

    # Ищем слой с фонемами. Если слой называется "phones" или "phonemes", выбираем его.
    tier = None
    for t in tg.tiers:
        if t.name.lower() in ['phone', 'phones', 'phoneme', 'phonemes']:
            tier = t
            break
    if tier is None:
        # Если специфичный слой не найден, берём первый слой
        tier = tg.tiers[0]

    phonemes = []
    pitches = []

    # Проходим по всем интервалам выбранного слоя
    for interval in tier.intervals:
        label = interval.mark.strip()
        # Если метка пустая, пропускаем
        if not label:
            continue
        tmin = interval.minTime
        tmax = interval.maxTime
        avg_pitch = get_average_pitch(pitch, tmin, tmax)
        phonemes.append(label)
        pitches.append(f"{avg_pitch:.2f}" if avg_pitch is not None else "N/A")
    return phonemes, pitches


def main():
    # Указываем базовые директории
    audio_base = "LibriTTS_R/test-clean"
    alignments_base = "test-clean-alignments"
    output_csv = "markup_labels.csv"

    all_results = []  # Собираем результаты из всех файлов

    # Открываем CSV файл для записи результатов


    # Итерируем по папкам. Предполагается, что структура такая:
    # LibriTTS_R/test-clean/{speaker}/{subfolder}/{utterance}.wav
    # test-clean-alignments/{speaker}/{utterance}.TextGrid
    for speaker in os.listdir(audio_base):
        speaker_path = os.path.join(audio_base, speaker)
        if not os.path.isdir(speaker_path):
            continue
        for subfolder in os.listdir(speaker_path):
            subfolder_path = os.path.join(speaker_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for file in os.listdir(subfolder_path):
                if file.endswith(".wav"):
                    wav_path = os.path.join(subfolder_path, file)
                    # Формируем имя TextGrid (предполагается, что имя файла совпадает)
                    textgrid_filename = os.path.splitext(file)[0] + ".TextGrid"
                    textgrid_path = os.path.join(alignments_base, speaker, textgrid_filename)
                    if not os.path.exists(textgrid_path):
                        # print(f"TextGrid не найден для {wav_path}")
                        continue
                    # print(f"\nОбработка файла:\n  WAV: {wav_path}\n  TextGrid: {textgrid_path}")
                    phonemes, pitches = process_utterance(wav_path, textgrid_path)

                    if phonemes:
                        all_results.append([file, " ".join(phonemes), " ".join(pitches)])
                    # for res in results:
                        # all_results.append((wav_path,) + res)
                    # Выводим результаты
                    # for label, tmin, tmax, avg_pitch in results:
                        # pitch_str = f"{avg_pitch:.2f} Hz" if avg_pitch is not None else "N/A"
                        # print(f"Фонема: {label:5s} | Интервал: {tmin:6.3f}-{tmax:6.3f} сек | Средний питч: {pitch_str}")
                    # print("-" * 60)

    # Записываем все результаты в CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Файл", "Фонемы", "Средние питчи (Hz)"])
        writer.writerows(all_results)
        # for file_path, label, tmin, tmax, avg_pitch in all_results:
            # pitch_str = f"{avg_pitch:.2f}" if avg_pitch is not None else "N/A"
            # writer.writerow([file_path, label, f"{tmin:.3f}", f"{tmax:.3f}", pitch_str])

    print(f"\nРезультаты сохранены в файле: {output_csv}")


if __name__ == "__main__":
    main()
