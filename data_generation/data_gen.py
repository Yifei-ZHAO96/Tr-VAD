import argparse
from add_silence import add_silence
from augmentation import augmentation
import multiprocessing
from functools import partial
from augmentation import find_files
from tqdm import tqdm


def list_of_ints(string):
    """Convert a string representation of a list of integers to an actual list."""
    try:
        return [int(x) for x in string[1:-1].split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid list of integers: {string}")


def parallelize_augmentation(clean_file_paths, noise_file_path, sr, snr_):
    pool = multiprocessing.Pool()
    func = partial(augmentation, noise_file_path=noise_file_path, sr=sr, snr_=snr_)
    total = len(clean_file_paths)

    with tqdm(total=total, unit='files') as pbar:
        for _ in pool.imap_unordered(func, clean_file_paths):
            pbar.update()

    pool.close()
    pool.join()

if __name__ == '__main__':
    """
    Use example: 
        python data_gen.py '.../TIMIT/TRAIN' '.../AURORA' -sr 16000 -silence_pad 1 -snr "[-5, 0, 5, 10]"
    """
    parser = argparse.ArgumentParser(
        description="Data generation for TIMIT dataset.")
    parser.add_argument("clean_data_path",
                        help="Input clean file path")
    parser.add_argument("noise_data_path",
                        help="Input noise file path")
    # optional args
    parser.add_argument("-sr", "--sample_rate", type=int,
                        default=16000, help="Sampling rate")
    parser.add_argument("-silence_pad", "--silence_padding", type=int,
                        default=1, help="Silence padding duration in second")
    parser.add_argument("-snr", "--snr", type=list_of_ints, default=[-10, -5, 0, 5, 10], help="List of integers representing snrs")

    args = parser.parse_args()

    print('Augmented data path is: {}'.format(
        args.clean_data_path.replace('TIMIT', 'TIMIT_augmented')))

    add_silence(clean_file_path=args.clean_data_path,
                sr=args.sample_rate,
                sil_length=args.silence_padding)
    
    clean_file_path = args.clean_data_path.replace('TIMIT', 'TIMIT_augmented')
    clean_file_paths = find_files(clean_file_path, '**/*.WAV') or find_files(clean_file_path, '**/*.WAV')
    clean_file_paths = [f.replace('\\', '/') for f in clean_file_paths]
    noise_path = find_files(args.noise_data_path, '**/*.wav') or find_files(args.noise_data_path, '**/*.WAV')
    noise_files = [path.replace('\\', '/') for path in noise_path]
    
    print('Adding noises...')
    parallelize_augmentation(clean_file_paths, noise_files, sr=args.sample_rate, snr_=args.snr)
    print('Noises added.')
