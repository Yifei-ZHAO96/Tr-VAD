import numpy as np

from AFPC_feature import base


def features(test_noisy, fs=16000, nfft=512, winlen=0.032, winstep=0.016, nfilt=64, ncoef=22):

    # Extract MFCC
    mfcc = base.mfcc(signal=test_noisy,
                     samplerate=fs,
                     winlen=winlen,
                     winstep=winstep,
                     numcep=ncoef,
                     nfilt=nfilt,
                     nfft=nfft,
                     lowfreq=0,
                     highfreq=8000,
                     preemph=0.97,
                     ceplifter=ncoef,
                     appendEnergy=False)
    # Extract NSSC
    ssc = base.ssc(signal=test_noisy,
                   samplerate=fs,
                   winlen=winlen,
                   winstep=winstep,
                   nfilt=nfilt,
                   nfft=nfft,
                   lowfreq=0,
                   highfreq=8000,
                   preemph=0.97)
    nssc = base.norm_ssc(ssc,
                         nfilt=nfilt,
                         nfft=nfft,
                         samplerate=fs,
                         lowfreq=0,
                         highfreq=8000)
    nssc = nssc[:, 0:ncoef]

    # Delta NSSCs
    delta_ssc = base.delta(nssc, 2)
    delta2_ssc = base.delta(delta_ssc, 2)
    nssc_pac = np.concatenate((nssc, delta_ssc, delta2_ssc), axis=1)

    # Delta MFCCs
    delta_mfcc = base.delta(mfcc, 2)
    delta2_mfcc = base.delta(delta_mfcc, 2)
    mfcc_pac = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=1)

    # AFPCS
    AFPC = np.concatenate((mfcc_pac, nssc_pac), axis=1).astype('float32')
    return AFPC
