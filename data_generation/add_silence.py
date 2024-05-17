# -*- coding: utf-8 -*-
import numpy as np
import librosa
from tqdm import tqdm
from shutil import copyfile
from scipy.io import wavfile
from glob import glob
import os


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def find_files(directory, pattern='**/*.WAV'):
    return sorted(glob(os.path.join(directory, pattern), recursive=True))

def add_silence(clean_file_path: str, sr=16000, sil_length=1, eps=0.00005):
    """
    Add silence padding to the audio files.

    Args:
        clean_file_path (str): clean file path.
        sr (int, optional): Sampling rate. Defaults to 16000.
        sil_length (int, optional): silence padding length. Defaults to 1.
        eps (float, optional): A very small value to prevent from dividing zero. Defaults to 0.00005.
    """
    print(f'Adding {sil_length}s silence...')
    for clean_file in tqdm(find_files(clean_file_path, '**/*.WAV') or find_files(clean_file_path, '**/*.wav')):
        
        file_title_ = clean_file.split("/")[-1].split('.')[0]
        clean_amp = load_wav(clean_file, sr=sr)
        rand_num = np.random.rand(sr * sil_length) * 2 - 1
        clean_add_sil_amp = np.concatenate(
            (rand_num * eps, clean_amp, rand_num * eps))
        output_dir = clean_file.replace('TIMIT', 'TIMIT_augmented').split('/')[:-1]
        output_dir = '/'.join(output_dir)
        output_file = output_dir + '/' + file_title_ + '_add_sil' + '.WAV'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_wav(clean_add_sil_amp, output_file, sr)

        # copy .PHN files
        parts_phn = clean_file.split('/')[:-1]
        input_phn = '/'.join(parts_phn) + '/' + file_title_ + '.PHN'
        output_phn = output_dir + '/' + file_title_ + '_add_sil' + '.PHN'
        copyfile(input_phn, output_phn)

    print(f'Added {sil_length}s silence...')
