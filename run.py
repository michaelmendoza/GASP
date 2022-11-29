import numpy as np
import matplotlib.pyplot as plt 
from gasp import dataloader, view


filepath = '../data/2019_GASP/20190507_GASP_LONG_TR_WATER/'
files = ['meas_MID12_TRUFI_TE12_FID42712.dat',
         'meas_MID13_TRUFI_TE24_FID42713.dat',
         'meas_MID14_TRUFI_TE48_FID42714.dat']

M0 = dataloader.read_rawdata(filepath + files[0], doChaSOSAverage=True)['data']
M1 = dataloader.read_rawdata(filepath + files[1], doChaSOSAverage=True)['data']
M2 = dataloader.read_rawdata(filepath + files[2], doChaSOSAverage=True)['data']