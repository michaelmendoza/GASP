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

def load_dataset3():
    url = 'https://drive.google.com/file/d/1k4NIhvuCBZKDNKcF4aFhtXEL2JbKw8VU/view?usp=sharing'
    dataloader.download_data(url,'20190506_GASP_CRISCO_WATER_PHANTOMS')
    filepath =  os.path.join(os.getcwd(), 'data', '20190506_GASP_CRISCO_WATER_PHANTOMS', '') 
    files = ['meas_MID22_TRUFI_TE12_FID42700.dat',
             'meas_MID24_TRUFI_TE24_FID42702.dat',
             'meas_MID25_TRUFI_TE48_FID42703.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset4():
    url = 'https://drive.google.com/file/d/1yxmriHAoNubNtyZDPak-TmM29tBHDmFv/view?usp=sharing'
    dataloader.download_data(url,'20190508_GASP_WATER_FAT_PHANTOM')
    filepath =  os.path.join(os.getcwd(), 'data', '20190508_GASP_WATER_FAT_PHANTOM', '') 
    files = ['meas_MID28_TRUFI_TE3_FID42728.dat',
             'meas_MID31_TRUFI_TE6_FID42731.dat',
             'meas_MID33_TRUFI_TE12_FID42733.dat']
    
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset5():
    url = 'https://drive.google.com/file/d/1O8xm9yWk-3vA8H90d3bVer7Ec8u143G-/view?usp=sharing'
    dataloader.download_data(url, '20190812_GASP_INVIVO_Sag_Knee')
    filepath =  os.path.join(os.getcwd(), 'data', '20190812_GASP_INVIVO_Sag_Knee', '') 
    files = ['meas_MID131_TRUFI_TE3_FID48578.dat', 
             'meas_MID132_TRUFI_TE6_FID48579.dat',
             'meas_MID133_TRUFI_TE12_FID48580.dat']
    
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset6a():
    url = 'https://drive.google.com/file/d/18NV--KkY9QmXm9OVSL73Lvm8Iqsw7ALH/view?usp=sharing'
    dataloader.download_data(url, '20190827_GASP_INVIVO_BRAIN_HIP')
    filepath =  os.path.join(os.getcwd(), 'data', '20190827_GASP_INVIVO_BRAIN_HIP', '') 
    files = ['meas_MID299_TRUFI_TE3_FID49324.dat',
             'meas_MID300_TRUFI_TE6_FID49325.dat',
             'meas_MID301_TRUFI_TE12_FID49326.dat']
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M
    
def load_dataset7a():
    filepath =  os.path.join(os.getcwd(), 'data', '20231106_GASP_PHANTOM', '') 
    files = ['meas_MID162_bSSFP_gasp_knee_fa90_1x1x2_2D_TR6ms_FID55595.dat',
             'meas_MID163_bSSFP_gasp_knee_fa90_1x1x2_2D_TR12ms_FID55596.dat',
             'meas_MID164_bSSFP_gasp_knee_fa90_1x1x2_2D_TR24ms_FID55597.dat']    
    
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset7b():
    filepath =  os.path.join(os.getcwd(), 'data', '20231106_GASP_PHANTOM', '') 
    files = ['meas_MID165_bSSFP_gasp_knee_fa20_1x1x2_2D_TR6ms_FID55598.dat',
             'meas_MID166_bSSFP_gasp_knee_fa20_1x1x2_2D_TR12ms_FID55599.dat',
             'meas_MID167_bSSFP_gasp_knee_fa20_1x1x2_2D_TR24ms_FID55600.dat']    
    
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset8a():
    filepath =  os.path.join(os.getcwd(), 'data', '20231106_GASP_KNEE', '') 
    files = ['meas_MID123_bSSFP_gasp_knee_faMax_1x1x2_2D_TR6ms_FID55556.dat',
             'meas_MID124_bSSFP_gasp_knee_faMax_1x1x2_2D_TR12ms_FID55557.dat',
             'meas_MID125_bSSFP_gasp_knee_faMax_1x1x2_2D_TR24ms_FID55558.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset8b():
    filepath =  os.path.join(os.getcwd(), 'data', '20231106_GASP_KNEE', '') 
    files = ['meas_MID127_bSSFP_gasp_knee_fa20_1x1x2_2D_TR6ms_FID55560.dat',
             'meas_MID128_bSSFP_gasp_knee_fa20_1x1x2_2D_TR12ms_FID55561.dat',
             'meas_MID129_bSSFP_gasp_knee_fa20_1x1x2_2D_TR24ms_FID55562.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset8c():
    filepath =  os.path.join(os.getcwd(), 'data', '20231106_GASP_KNEE', '') 
    files = ['meas_MID126_DIXON_2D_3echoes_FID55559.dat']
    M = dataloader.read_rawdata(filepath + files[0])['data']
    return M