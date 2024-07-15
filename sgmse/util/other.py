import numpy as np
import scipy.stats
import torch
import csv
import os
import glob
import tqdm
import matplotlib.pyplot as plt
import time
import scipy.signal as ss

stft_kwargs = {"n_fft": 510, "hop_length": 128, "window": torch.hann_window(510), "return_complex": True}

def si_sdr_components(s_hat, s, n, eps=1e-10):
    # s_target
    alpha_s = np.dot(s_hat, s) / (eps + np.linalg.norm(s)**2)
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / (eps + np.linalg.norm(n)**2)
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise
    
    return s_target, e_noise, e_art

def energy_ratios(s_hat, s, n, eps=1e-10):
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_noise + e_art)**2))
    si_sir = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_noise)**2))
    si_sar = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_art)**2))

    return si_sdr, si_sir, si_sar

def mean_conf_int(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

class Method():
    def __init__(self, name, base_dir, metrics):
        self.name = name
        self.base_dir = base_dir
        self.metrics = {} 
        
        for i in range(len(metrics)):
            metric = metrics[i]
            value = []
            self.metrics[metric] = value 
            
    def append(self, matric, value):
        self.metrics[matric].append(value)

    def get_mean_ci(self, metric):
        return mean_conf_int(np.array(self.metrics[metric]))


def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr


def pad_spec(Y):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    return pad2d(Y)
