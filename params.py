import torch


class HParams:
    def __init__(self):
        # FFT Params
        self.n_fft = 512
        self.winlen = 0.032
        self.winstep = 0.016
        self.nfilt = 64
        self.ncoef = 16

        self.sample_rate = 16000
        self.w = 16
        self.u = 4
        self.L = 9

        # model parameters
        self.dim_in = 80
        self.d_model = 162
        self.units_in = 9
        self.units = 54
        self.layers = 6
        self.P = 18
        self.drop_rate = 0.15
        self.activation = torch.nn.GELU()
        self.batch_size = 512
        self.epochs = 10

        self.lr = 1e-3
        self.warmup_lr_init = 5e-7
        self.lr_min = 5e-6
        self.weight_decay = 5e-2
        self.n_iter_step = 200_000
        self.warmup_factor = 0.05

        self.vad_reg_weight = 1e-6
