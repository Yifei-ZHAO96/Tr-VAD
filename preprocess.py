import numpy as np
import torch
import torchaudio
from AFPC_feature.AFPC import features
from utils import HParams
from tqdm import tqdm
from glob import glob
from functools import partial
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


hparams = HParams()
FS = hparams.sample_rate
NFFT = hparams.n_fft
WINSTEP = hparams.winstep
WINLEN = hparams.winlen
NFILT = hparams.nfilt
NCOEF = hparams.ncoef


def find_files(directory, pattern='**/*.WAV'):
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


def preprocess_parallel(input_dir: str, feature_dir: str, n_jobs=12, tqdm=lambda x: x, silence=1) -> list:
    """
    Prepare training/testing dataset.

    Args:
        input_dir (str): Input data directory.
        feature_dir (str): Input data save directory after transformation.
        n_jobs (int, optional): Number of jobs. Defaults to 12.
        tqdm (_type_, optional): tqdm function. Defaults to lambda x: x.
        silence (int, optional): Silence duration. Defaults to 1.

    Returns:
        list: output metadata including (wav_path, feature_filename, time_steps, num_frames, vad_start_time_stamp, vad_end_time_stamp)
    """
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1

    files = find_files(os.path.join(input_dir))
    print('number of files: ', len(files))
    for wav_path in files:
        text_path = os.path.splitext(wav_path)[0]
        text_parts = text_path.split('/')
        text_path = '/'.join(text_parts) + '.PHN'

        with open(text_path, encoding='utf-8') as f:
            lines = f.readlines()
            start = int(lines[1].split(' ')[0])
            end = int(lines[-2].split(' ')[1])
            futures.append(executor.submit(
                partial(_process_AFPC, feature_dir, index, wav_path, start, end, silence)))
            index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_AFPC(feature_dir, index, wav_path, start, end, silence):

    try:
        # Load the audio as numpy array
        wav, _ = torchaudio.load(wav_path)
        wav = wav[0]
    except FileNotFoundError:  # catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    start += int(silence * FS)
    end += int(silence * FS)

    # rescale wav
    wav = wav / torch.abs(wav).max() * 0.999
    out = wav.detach().numpy()

    feature_input = features(out, fs=FS, nfft=NFFT, winstep=WINSTEP, winlen=WINLEN,
                             nfilt=NFILT, ncoef=NCOEF)[:, :80]

    num_frames = feature_input.shape[0]

    out = np.pad(out, (0, NFFT // 2), mode='reflect')
    out = out[:num_frames * int(WINSTEP * FS)]
    time_steps = len(out)

    start = round(start / int(time_steps / num_frames))
    end = round(end / int(time_steps / num_frames))

    feature_filename = 'afpc-{}.npy'.format(index)
    np.save(os.path.join(feature_dir, feature_filename),
            feature_input, allow_pickle=False)

    # Return a tuple describing this training example
    return (wav_path, feature_filename, time_steps, num_frames, start, end)


def write_metadata(hparams, metadata, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    timesteps = sum([int(m[2]) for m in metadata])
    sr = hparams.sample_rate
    hours = timesteps / sr / 3600
    print('Write {} utterances, {} audio timesteps, ({:.2f} hours)'.format(
        len(metadata), timesteps, hours))
    print('Max audio timesteps length: {:.2f} secs'.format(
        (max(m[2] for m in metadata)) / sr, ))


def preprocess(input_folders: str, output_folder: str, output_name: str = 'list.txt', n_jobs=cpu_count(), silence=1):
    """
    Prepare dataset.

    Args:
        input_folders (str): input data folders.
        output_folder (str): output feature folder.
        output_name (str, optional): output summary filename. Defaults to 'list.txt'.
        n_jobs (int, optional): number of jobs. Defaults to cpu_count().
        silence (int, optional): silence duration. Defaults to 1.
    """
    feature_dir = os.path.join(output_folder, 'afpc')
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)

    metadata = preprocess_parallel(input_folders, feature_dir,
                                   n_jobs=n_jobs, tqdm=tqdm, silence=silence)
    write_metadata(hparams, metadata, os.path.join(output_folder, output_name))


if __name__ == '__main__':
    """
    Use example: 
        python preprocess.py '.../TIMIT_augmented/TRAIN' -silence_pad 1"
    """
    parser = argparse.ArgumentParser(
        description="Data generation for TIMIT dataset.")
    parser.add_argument("input_folder",
                        help="Path to the augmented dataset, e.g., <path_to_TIMIT_augmented>")
    parser.add_argument("-silence_pad", "--silence_padding", type=int,
                        default=1, help="Silence padding duration in second")

    args = parser.parse_args()

    silence = args.silence_padding
    input_folder = args.input_folder
    output_folder = input_folder.split('/')[:-1]
    output_folder = '/'.join(output_folder) + '/train'

    preprocess(input_folder, output_folder, output_name='list.txt',
               n_jobs=cpu_count()-1, silence=silence)
    print('Done')
