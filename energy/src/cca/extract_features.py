import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

resample_rate = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_fbank(wavform,
                  sample_rate=16000,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  cmn=True):
    feat = kaldi.fbank(wavform,
                       num_mel_bins=num_mel_bins,
                       frame_length=frame_length,
                       frame_shift=frame_shift,
                       sample_frequency=sample_rate)
    if cmn:
        feat = feat - torch.mean(feat, 0)
    return feat


def extract_features(audio_path: str, start_time: float = None, end_time: float = None):
    pcm, sample_rate = torchaudio.load(audio_path, normalize=False)

    if start_time is not None and end_time is not None:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        pcm = pcm[:, start_sample:end_sample]

    return extract_feature_from_pcm(pcm, sample_rate)


def extract_feature_from_pcm(pcm: torch.Tensor, sample_rate: int):
    pcm = pcm.to(torch.float)
    if sample_rate != resample_rate:
        pcm = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(pcm)
    feats = compute_fbank(pcm,
                          sample_rate=resample_rate,
                          cmn=True)
    feats = feats.unsqueeze(0)
    feats = feats.to(device)

    return feats
