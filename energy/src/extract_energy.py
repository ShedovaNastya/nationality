import os
from pathlib import Path
import pandas as pd
import numpy as np
import parselmouth

import textgrid


def extract_phoneme_energy(textgrid_path, wav_path):
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    sound = parselmouth.Sound(wav_path)

    phoneme_tier = None
    for tier in tg:
        if "phones" in tier.name.lower():
            phoneme_tier = tier
            break

    if phoneme_tier is None:
        raise ValueError(f"no phoneme tier in {textgrid_path}")

    data = []

    for interval in phoneme_tier:
        phoneme = interval.mark.strip()
        if phoneme == "":
            continue

        start, end = interval.minTime, interval.maxTime
        segment_duration = end - start

        if segment_duration < 0.064:
            continue

        segment = sound.extract_part(from_time=start, to_time=end, preserve_times=True)

        intensity = segment.to_intensity(time_step=0.01)
        energy = np.mean(intensity.values)

        data.append([phoneme, energy, wav_path, start, end])

    return data


def get_audio_paths(audio_dir):
    """
    Recursively finds all audio files in the specified directory.
    """
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob('*.TextGrid'))
    if audio_files.__len__() == 0:
        for path in os.listdir(audio_dir):
            audio_files.append(get_audio_paths(os.path.join(audio_dir, path)))

    return audio_files


def process_dataset(textgrid_dir, wav_dir, output_csv):
    results = []

    for filename in os.listdir(textgrid_dir):
        files = get_audio_paths(os.path.join(textgrid_dir, filename))
        for file in files:
            if file.__str__().endswith(".TextGrid"):
                wav_filename = os.readlink(file.__str__().replace(".TextGrid", ".wav"))
                wav_path = wav_filename

                if os.path.exists(wav_path):
                    results.extend(extract_phoneme_energy(file.__str__(), wav_path.__str__()))
                else:
                    print(f"no wav files found in {wav_filename}")
                    break

    df = pd.DataFrame(results, columns=["phoneme", "energy", "wav_path", "start_time", "end_time"])
    df.to_csv(output_csv, index=False)
    print(f"file saved: {output_csv}")


if __name__ == "__main__":
    TEXTGRID_DIR = "../data/test-clean-alignments"
    WAV_DIR = "../data/test-clean"
    OUTPUT_CSV = "../data/phoneme_energy.csv"

    process_dataset(TEXTGRID_DIR, WAV_DIR, OUTPUT_CSV)

