import librosa
import os
import matplotlib.pyplot as plt
from utils import bdnn_prediction, get_parameter_number, data_transform
from params import HParams
from VAD_T import VADModel
import numpy as np
from AFPC_feature import AFPC
import torch.nn.functional as F
import torch
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Function to convert frame-level VAD output to sample-level
def frame2sample(label, w_len, w_step):
    num_frame = len(label)
    total_len = (num_frame - 1) * w_step + w_len
    raw_label = np.zeros(total_len)
    index = 0
    i = 0

    while True:
        if index + w_len >= total_len:
            break
        if i ==0:
            raw_label[index : index+w_len] = label[i]
        else:
            temp_label = label[i]
            raw_label[index: index+w_len] += temp_label

        i += 1
        index += w_step

    raw_label[raw_label >= 1] = 1
    raw_label[raw_label < 1] = 0

    return raw_label


def parse_args():
    parser = argparse.ArgumentParser(description='Speech VAD Inference')
    parser.add_argument('--input_path', type=str, 
                        default='./data_test/[NOISE]SA1_add_sil_SNR(00)_airport.WAV',
                        help='Path to the input audio file')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/weights_10_acc_97.09.pth',
                        help='Path to the checkpoint file')
    return parser.parse_args()


def main():
    args = parse_args()
    hparams = HParams()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_path = args.input_path
    print(f'input_path: {input_path}')

    checkpoint_path = args.checkpoint_path
    print(f'checkpoint_path: {checkpoint_path}')

    model = VADModel(dim_in=hparams.dim_in, d_model=hparams.d_model, units_in=hparams.units_in,
                     units=hparams.units, layers=hparams.layers, P=hparams.P, drop_rate=0,
                     activation=hparams.activation).to(DEVICE)
    get_parameter_number(model)
    window_size, unit_size = hparams.w, hparams.u

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    waveform, sr = librosa.load(input_path, sr=hparams.sample_rate)
    waveform = waveform / np.abs(waveform).max() * 0.999

    feature_input = AFPC.features(waveform, fs=sr, nfft=hparams.n_fft, winstep=hparams.winstep,
                                  winlen=hparams.winlen, nfilt=hparams.nfilt, ncoef=hparams.ncoef)[:, :80]
    feature_input = (feature_input - np.mean(feature_input, axis=0)
                     ) / (np.std(feature_input, axis=0) + 1e-10)
    feature_input = torch.as_tensor(feature_input, dtype=torch.float32)
    feature_input = data_transform(
        feature_input, window_size, unit_size, feature_input.min(), DEVICE=torch.device('cpu'))
    feature_input = feature_input[window_size: -window_size, :, :]

    with torch.inference_mode():
        train_data = feature_input.to(DEVICE)
        postnet_output = model(train_data)
        _, vad = bdnn_prediction(F.sigmoid(postnet_output).cpu(
        ).detach().numpy(), w=window_size, u=unit_size, threshold=0.4)

    wav_out = np.pad(waveform, (0, hparams.n_fft // 2), mode='reflect')
    vad = np.concatenate((np.zeros(hparams.w), vad[:, 0], np.zeros(hparams.w)))
    vad_sample = frame2sample(vad, int(hparams.sample_rate * hparams.winlen), int(hparams.sample_rate * hparams.winstep))

    plt.plot(wav_out)
    plt.plot(vad_sample)
    plt.savefig('{}'.format('.' + input_path.split('.')[1] + '.png'))


if __name__ == '__main__':
    main()
