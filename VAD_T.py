import torch
import copy
import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F

from params import HParams
import numpy as np
import math


hparamas = HParams()


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Embedding(nn.Module):
    def __init__(self, dim_in=80, dim_out=128, units_in=9, units_out=18, drop_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out * 2),
            nn.Conv1d(units_in, units_out, kernel_size=5, stride=2, padding=2),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        return x


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                      padding=padding, groups=dim_in, stride=stride, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, units=16, dim=128, P=16, ratio=4, drop_rate=0.3, activation=nn.ReLU()):
        super().__init__()
        dim_in = int(units * dim / P ** 2)
        self.P = P
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * ratio, 1),
            activation,
            DepthWiseConv2d(dim_in * ratio, dim_in * ratio, 3, 1),
            activation,
            nn.Dropout(drop_rate),
            nn.Conv2d(dim_in * ratio, dim_in, 1),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        c = int(x.shape[1] / self.P)
        x = rearrange(x, 'b (c p1) (d p2)-> b (c d) p1 p2',
                      p1=self.P, p2=self.P)
        x = self.net(x)
        x = rearrange(x, 'b (c d) p1 p2 -> b (c p1) (d p2)', c=c)
        return x


class Encoder(nn.Module):
    def __init__(self, h=8, d_model=392, units=16, P=16, drop_rate=0.3, layers=2, activation=nn.ReLU()):
        super().__init__()
        self.att = clones(
            MultiHeaded(h=h, units=units, d_model=d_model, drop_rate=drop_rate, P=P, activation=activation), layers)
        self.ffn = clones(
            FeedForward(units=units, dim=d_model, P=P, ratio=4, drop_rate=drop_rate, activation=activation), layers)
        self.norm1 = clones(nn.LayerNorm(d_model), layers)
        self.norm2 = clones(nn.LayerNorm(d_model), layers)

    def forward(self, x):
        for att, ffn, norm1, norm2 in zip(self.att, self.ffn, self.norm1, self.norm2):
            x1 = norm1(x)
            x = x + att(x1, x1, x1)
            x2 = norm2(x)
            x = x + ffn(x2)
        return x


def attention(query, key, value, mask=None, dropout=None, position=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    if position is not None:
        scores += position.unsqueeze(0)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)

    return output, p_attn


class MultiHeaded(nn.Module):
    def __init__(self, h=8, units=16, d_model=392, drop_rate=0.3, P=16, activation=nn.ReLU()):
        "Take in model size and number of heads."
        super(MultiHeaded, self).__init__()
        self.P = P
        self.h = h
        dim_in = int(units * d_model / P ** 2)
        self.wq = DepthWiseConv2d(dim_in, dim_in, 3, 1, 2, False)
        self.wk = DepthWiseConv2d(dim_in, dim_in, 3, 1, 2, False)
        self.wv = DepthWiseConv2d(dim_in, dim_in, 3, 1, 2, False)
        self.attn = None
        self.dropout = nn.Dropout(p=drop_rate)
        self.activation = activation
        self.position = nn.Parameter(
            torch.randn(units//P, d_model//P, d_model//P))
        # self.position = None
        self.linear = nn.Sequential(
            nn.Conv1d(units//2, units, 1),
            nn.Linear(d_model // 2, d_model),
            nn.Dropout(drop_rate)
        )

    def forward(self, query, key, value, mask=None):
        c = key.shape[1] // self.P
        query, key, value = [rearrange(x, 'b (c p1) (d p2)-> b (c d) p1 p2', p1=self.P, p2=self.P) for x in
                             (query, key, value)]

        if mask is not None:
            mask = mask.unsqueeze(1)
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        query = rearrange(query, 'b (c d) p1 p2 -> b c d (p1 p2)', c=c)
        key = rearrange(key, 'b (c d) p1 p2 -> b c d (p1 p2)', c=c)
        value = rearrange(value, 'b (c d) p1 p2 -> b c d (p1 p2)', c=c)

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout, position=self.position)

        x = rearrange(x, 'b c d (p1 p2) -> b (c p1) (d p2)', p1=self.P // 2)
        x = self.linear(x)
        return x


class Postnet(nn.Module):
    def __init__(self, dim_in=56, ratio=2, units=16, P=16, drop_rate=0.3, activation=nn.ReLU()):
        super().__init__()
        c_units = int(dim_in*units/P**2)
        self.P = P
        self.conv = nn.Sequential(
            DepthWiseConv2d(c_units, c_units, 5, 2, 2),
        )
        # self.conv = nn.Conv2d(c_units, c_units//2, 7, 3, 1)
        c_linear = dim_in*units//(1*2**2 * P//2)
        self.net = nn.Sequential(
            nn.Linear(c_linear, c_linear*ratio),
            nn.BatchNorm1d(P//2),
            activation,
            nn.Dropout(drop_rate),
            nn.Linear(c_linear*ratio, 1),
        )
        self.norm = nn.LayerNorm(dim_in)

    def forward(self, x):
        x = self.norm(x)
        x = rearrange(x, 'b (c p1) (d p2)-> b (c d) p1 p2',
                      p1=self.P, p2=self.P)
        x = self.conv(x)
        x = rearrange(x, 'b c p1 p2-> b p1 (c p2)')
        x = self.net(x)
        return torch.squeeze(x)


class VADModel(nn.Module):
    def __init__(self, dim_in=80, d_model=56, units_in=8, units=16, P=16, layers=2, drop_rate=0.3,
                 activation=nn.ReLU()):
        super().__init__()
        self.net = nn.Sequential(
            Embedding(dim_in=dim_in, dim_out=d_model, units_in=units_in,
                      units_out=units, drop_rate=drop_rate),
            Encoder(h=8, d_model=d_model, units=units, P=P,
                    drop_rate=drop_rate, layers=layers, activation=activation),
            Postnet(dim_in=d_model, ratio=2, units=units, P=P,
                    drop_rate=drop_rate, activation=activation)
        )

    def forward(self, x):
        x = self.net(x)
        return x


def add_loss(model, targets, post_out):
    post_loss = (F.binary_cross_entropy_with_logits(post_out, targets)).mean()
    regularization = sum([torch.norm(val, 2) for name, val in model.named_parameters() if
                          'weight' in name]).item() * hparamas.vad_reg_weight
    total_loss = post_loss + regularization

    return total_loss, post_loss


def prediction(targets, post_out, w=hparamas.w, u=hparamas.u):
    post_prediction = torch.round(F.sigmoid(post_out))
    post_acc = torch.mean(post_prediction.eq_(targets))

    return post_acc
