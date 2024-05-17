import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio


def _preemphasis(wav, k=0.97, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def stft(input, hparams):
    input = _preemphasis(input)
    input = torch.as_tensor(input, dtype=torch.float32)
    return torch.stft(input=input, n_fft=hparams.n_fft, hop_length=hparams.hop_length,
                      win_length=hparams.win_size, return_complex=False,
                      window=torch.hann_window(hparams.win_size), normalized=True)

def melspectrogram(wav, hparams):
    amp2dB = torchaudio.transforms.AmplitudeToDB()
    mel = torchaudio.transforms.MelScale(sample_rate=hparams.sample_rate, n_mels=hparams.num_mels,
                                         f_min=0, f_max=8000, n_stft=hparams.n_fft // 2 + 1)

    wav = stft(wav, hparams)
    wav = mel(wav.permute([1, 0, 2]))
    return amp2dB(wav)
