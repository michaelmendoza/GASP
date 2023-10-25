import os
import numpy as np
import matplotlib.pyplot as plt 
from gasp import dataloader

def load_dataset0():
    url = 'https://drive.google.com/file/d/1ttvKQUAPjRJdtocAIJzUv1_7n3kBEOSI/view?usp=share_link'
    dataloader.download_data(url, '20190401_GASP_PHANTOM')
    filepath =  os.path.join(os.getcwd(), 'data', '20190401_GASP_PHANTOM', '')   
    files = ['meas_MID48_TRUFI_NBPM_2019_02_27_FID41503.dat',
            'meas_MID49_TRUFI_NBPM_2019_02_27_FID41504.dat',
            'meas_MID50_TRUFI_NBPM_2019_02_27_FID41505.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset1():
    url = 'https://drive.google.com/file/d/1ttvKQUAPjRJdtocAIJzUv1_7n3kBEOSI/view?usp=share_link'
    dataloader.download_data(url, '20190401_GASP_PHANTOM')
    filepath =  os.path.join(os.getcwd(), 'data', '20190401_GASP_PHANTOM', '')   
    files = ['meas_MID54_TRUFI_NBPM_2019_02_27_FID41509.dat',
            'meas_MID55_TRUFI_NBPM_2019_02_27_FID41510.dat',
            'meas_MID56_TRUFI_NBPM_2019_02_27_FID41511.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset2():
    url = 'https://drive.google.com/file/d/1l-JqXUnn7WVubMRaSI1uVsenuQg1MUeS/view?usp=share_link'
    dataloader.download_data(url, '20190507_GASP_LONG_TR_WATER')
    filepath =  os.path.join(os.getcwd(), 'data', '20190507_GASP_LONG_TR_WATER', '')   
    files = ['meas_MID12_TRUFI_TE12_FID42712.dat',
            'meas_MID13_TRUFI_TE24_FID42713.dat',
            'meas_MID14_TRUFI_TE48_FID42714.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M
