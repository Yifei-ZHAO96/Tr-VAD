import random
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path
from params import HParams


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU = torch.device('cpu')
hparams = HParams()


def data_transform_targets_bdnn(inputs, w=8, u=2, DEVICE=DEVICE):
    neighbors = torch.arange(-w, w + 1, u)

    pad_size = 2 * w + inputs.shape[0]
    pad_inputs = torch.zeros(pad_size).to(DEVICE)
    pad_inputs[0:inputs.shape[0]] = inputs

    trans_inputs = torch.vstack([torch.unsqueeze(torch.roll(pad_inputs, int(-1 * neighbors[i]), dims=0)
                                                 [0:inputs.shape[0]], dim=0) for i in range(neighbors.shape[0])])
    trans_inputs = trans_inputs.permute([1, 0])

    return trans_inputs


def data_transform(inputs, w=8, u=2, min_abs_value=1e-7, DEVICE=DEVICE):
    neighbors = torch.arange(-w, w + 1, u)

    pad_size = 2 * w + inputs.shape[0]
    pad_inputs = torch.ones((pad_size, inputs.shape[1])).to(
        DEVICE) * max(min_abs_value, 1e-7)
    pad_inputs[0: inputs.shape[0], :] = inputs

    trans_inputs = torch.vstack([torch.unsqueeze(torch.roll(pad_inputs, int(-1 * neighbors[i]), dims=0)
                                                 [0: inputs.shape[0], :], dim=0) for i in range(neighbors.shape[0])])
    trans_inputs = trans_inputs.permute([1, 0, 2])

    return trans_inputs


def bdnn_prediction(logits, threshold=0.5, w=8, u=2):
    bdnn_batch_size = int(logits.shape[0] + 2 * w)
    result = np.zeros((bdnn_batch_size, 1))
    indx = torch.arange(bdnn_batch_size) + 1
    indx = data_transform_targets_bdnn(indx, w, u, DEVICE=CPU)
    indx = indx[w:(bdnn_batch_size - w), :]
    indx_list = np.arange(w, bdnn_batch_size - w)

    for i in indx_list:
        indx_temp = np.where((indx - 1) == i)
        pred = logits[indx_temp]
        pred = np.sum(pred) / pred.shape[0]
        result[i] = pred

    result = result[w: -w]
    soft_result = np.float32(result)
    result = np.float32(result) >= threshold

    return result.astype(np.float32), soft_result


def prediction(targets, pipenet_output, postnet_output, w=hparams.w, u=hparams.u):
    pipenet_prediction = torch.round(F.sigmoid(pipenet_output))
    postnet_prediction = torch.round(F.sigmoid(postnet_output))

    pipenet_targets = targets.clone().detach()
    raw_indx = int(np.floor(int(2 * (w - 1) / u + 3) / 2))
    raw_labels = pipenet_targets[:, raw_indx]
    raw_labels = torch.reshape(raw_labels, shape=(-1, 1))

    postnet_accuracy = torch.mean(postnet_prediction.eq_(raw_labels))
    pipenet_accuracy = torch.mean(pipenet_prediction.eq_(raw_labels))
    return postnet_accuracy, pipenet_accuracy


class SpeechDataset(Dataset):
    def __init__(self, metadata_filename, metadata, hparams):
        super().__init__()
        self._hparams = hparams
        self._mel_dir = os.path.join(
            os.path.dirname(metadata_filename), 'afpc')
        self._metadata = metadata
        timesteps = sum([int(x[2]) for x in self._metadata])
        sr = hparams.sample_rate
        hours = timesteps / sr / 3600
        print('Loaded metadata for {} examples ({:.2f} hours)'.format(
            len(self._metadata), hours))
        self.len_ = len(self._metadata)

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        meta = self._metadata[index]
        start_frame = int(meta[4])
        end_frame = int(meta[5])

        feature_input = np.load(os.path.join(self._mel_dir, meta[1]))[:, :80]
        feature_input = (feature_input - np.mean(feature_input,
                         axis=0)) / (np.std(feature_input, axis=0) + 1e-10)
        target = np.asarray([0] * (len(feature_input)))
        target[start_frame: end_frame] = 1

        feature_input = torch.as_tensor(feature_input, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        feature_input = data_transform(
            feature_input, self._hparams.w, self._hparams.u, feature_input.min(), DEVICE=torch.device('cpu'))
        target = data_transform_targets_bdnn(
            target, self._hparams.w, self._hparams.u, DEVICE=torch.device('cpu'))
        feature_input = feature_input[self._hparams.w: -self._hparams.w, :, :]
        target = target[self._hparams.w: -self._hparams.w]
        return feature_input, target


def train_valid_split(metadata_filename, test_size=0.05, seed=0):
    with open(metadata_filename, encoding='utf-8') as f:
        data = [line.strip().split('|') for line in f]
        timesteps = sum([int(x[2]) for x in data])
        sr = 16000
        hours = timesteps / sr / 3600
        print('Loaded metadata for {} examples ({:.2f} hours)'.format(len(data), hours))
    random.seed(seed)
    training_idx = []
    validation_idx = []

    aug_list = [[idx, x] for idx, x in enumerate(
        data) if len(Path(x[0]).name.split('_')) > 3]
    snr_list = list(set([Path(x[0]).name.split('_')[3]
                    for idx, x in aug_list]))
    noise_list = list(set([Path(x[0]).name.split('_')[4]
                      for idx, x in aug_list]))
    clean_idx = list(range(len(data)))
    for idx, x in aug_list:
        clean_idx.remove(idx)

    random.shuffle(clean_idx, random.random)
    validation_split_idx = int(np.ceil(test_size * len(clean_idx)))
    training_idx += clean_idx[validation_split_idx:]
    validation_idx += clean_idx[0: validation_split_idx]

    meta_idx = {}
    for n in noise_list:
        meta_idx[n] = {}
        for s in snr_list:
            meta_idx[n][s] = [idx for idx, x in aug_list if Path(x[0]).name.split(
                '_')[4] == n and Path(x[0]).name.split('_')[3] == s]
            random.shuffle(meta_idx[n][s])
            validation_split_idx = int(
                np.ceil(test_size * len(meta_idx[n][s])))
            training_idx += meta_idx[n][s][validation_split_idx:]
            validation_idx += meta_idx[n][s][0: validation_split_idx]

    if bool(set(training_idx) & set(validation_idx)):
        raise ValueError('Training and validation data are overlapped!')

    random.shuffle(training_idx)
    random.shuffle(validation_idx)

    return data, training_idx, validation_idx


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total: {:.3f}Million, Trainable: {:.3f}Million'.format(
        total_num / 1_000_000, trainable_num / 1_000_000))
    # return {'Total': total_num, 'Trainable': trainable_num}


class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []
