import os
import numpy as np
import matplotlib.pyplot as plt 
from gasp import dataloader, get_project_path

def load_dataset(url, foldername, files = None, path = get_project_path()):
    dataloader.download_data(url, foldername, path)
    filepath =  os.path.join(path, 'data', foldername, '')       
    if files is None:
        files = os.listdir(filepath)
        files.sort()
    data_list = [dataloader.read_rawdata(os.path.join(filepath, file))['data'] for file in files]
    M = np.stack(data_list, axis=-1)
    return M

def load_dataset0(path = os.getcwd()):
    url = 'https://drive.google.com/file/d/1ttvKQUAPjRJdtocAIJzUv1_7n3kBEOSI/view?usp=share_link'
    dataloader.download_data(url, '20190401_GASP_PHANTOM', path)
    filepath =  os.path.join(path, 'data', '20190401_GASP_PHANTOM', '')   
    files = ['meas_MID48_TRUFI_NBPM_2019_02_27_FID41503.dat',
            'meas_MID49_TRUFI_NBPM_2019_02_27_FID41504.dat',
            'meas_MID50_TRUFI_NBPM_2019_02_27_FID41505.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset1(path = os.getcwd()):
    url = 'https://drive.google.com/file/d/1ttvKQUAPjRJdtocAIJzUv1_7n3kBEOSI/view?usp=share_link'
    dataloader.download_data(url, '20190401_GASP_PHANTOM', path)
    filepath =  os.path.join(path, 'data', '20190401_GASP_PHANTOM', '')   
    files = ['meas_MID54_TRUFI_NBPM_2019_02_27_FID41509.dat',
            'meas_MID55_TRUFI_NBPM_2019_02_27_FID41510.dat',
            'meas_MID56_TRUFI_NBPM_2019_02_27_FID41511.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset2(path = os.getcwd()):
    url = 'https://drive.google.com/file/d/1l-JqXUnn7WVubMRaSI1uVsenuQg1MUeS/view?usp=share_link'
    dataloader.download_data(url, '20190507_GASP_LONG_TR_WATER', path)
    filepath =  os.path.join(path, 'data', '20190507_GASP_LONG_TR_WATER', '')   
    files = ['meas_MID12_TRUFI_TE12_FID42712.dat',
            'meas_MID13_TRUFI_TE24_FID42713.dat',
            'meas_MID14_TRUFI_TE48_FID42714.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset3(path = os.getcwd()):
    url = 'https://drive.google.com/file/d/1k4NIhvuCBZKDNKcF4aFhtXEL2JbKw8VU/view?usp=sharing'
    dataloader.download_data(url,'20190506_GASP_CRISCO_WATER_PHANTOMS', path)
    filepath =  os.path.join(path, 'data', '20190506_GASP_CRISCO_WATER_PHANTOMS', '') 
    files = ['meas_MID22_TRUFI_TE12_FID42700.dat',
             'meas_MID24_TRUFI_TE24_FID42702.dat',
             'meas_MID25_TRUFI_TE48_FID42703.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset4(path = os.getcwd()):
    url = 'https://drive.google.com/file/d/1yxmriHAoNubNtyZDPak-TmM29tBHDmFv/view?usp=sharing'
    dataloader.download_data(url,'20190508_GASP_WATER_FAT_PHANTOM', path)
    filepath =  os.path.join(path, 'data', '20190508_GASP_WATER_FAT_PHANTOM', '') 
    files = ['meas_MID28_TRUFI_TE3_FID42728.dat',
             'meas_MID31_TRUFI_TE6_FID42731.dat',
             'meas_MID33_TRUFI_TE12_FID42733.dat']
    
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset5(path = os.getcwd()):
    url = 'https://drive.google.com/file/d/1O8xm9yWk-3vA8H90d3bVer7Ec8u143G-/view?usp=sharing'
    dataloader.download_data(url, '20190812_GASP_INVIVO_Sag_Knee', path)
    filepath =  os.path.join(path, 'data', '20190812_GASP_INVIVO_Sag_Knee', '') 
    files = ['meas_MID131_TRUFI_TE3_FID48578.dat', 
             'meas_MID132_TRUFI_TE6_FID48579.dat',
             'meas_MID133_TRUFI_TE12_FID48580.dat']
    
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset6a(path = os.getcwd()):
    url = 'https://drive.google.com/file/d/18NV--KkY9QmXm9OVSL73Lvm8Iqsw7ALH/view?usp=sharing'
    dataloader.download_data(url, '20190827_GASP_INVIVO_BRAIN_HIP', path)
    filepath =  os.path.join(path, 'data', '20190827_GASP_INVIVO_BRAIN_HIP', '') 
    files = ['meas_MID299_TRUFI_TE3_FID49324.dat',
             'meas_MID300_TRUFI_TE6_FID49325.dat',
             'meas_MID301_TRUFI_TE12_FID49326.dat']
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M
    
def load_dataset7a():
    url = 'https://drive.google.com/file/d/11szQZR8MPmT09zaM-lCSc4nUlNC4E-el/view?usp=sharing'
    foldername = '20231106_GASP_PHANTOM'
    files = ['meas_MID162_bSSFP_gasp_knee_fa90_1x1x2_2D_TR6ms_FID55595.dat',
            'meas_MID163_bSSFP_gasp_knee_fa90_1x1x2_2D_TR12ms_FID55596.dat',
            'meas_MID164_bSSFP_gasp_knee_fa90_1x1x2_2D_TR24ms_FID55597.dat']  
    return load_dataset(url, foldername, files)

def load_dataset7b():
    url = 'https://drive.google.com/file/d/11szQZR8MPmT09zaM-lCSc4nUlNC4E-el/view?usp=sharing'
    foldername = '20231106_GASP_PHANTOM'
    files = ['meas_MID165_bSSFP_gasp_knee_fa20_1x1x2_2D_TR6ms_FID55598.dat',
            'meas_MID166_bSSFP_gasp_knee_fa20_1x1x2_2D_TR12ms_FID55599.dat',
            'meas_MID167_bSSFP_gasp_knee_fa20_1x1x2_2D_TR24ms_FID55600.dat']  
    return load_dataset(url, foldername, files)

def load_dataset8a(path = os.getcwd()):
    url = 'https://drive.google.com/file/d/1pbn2gsxe-PjvW9vSIE6pnfbCUVWrUGbe/view?usp=sharing'
    dataloader.download_data(url, '20231106_GASP_KNEE', path)
    filepath =  os.path.join(path, 'data', '20231106_GASP_KNEE', '') 
    files = ['meas_MID123_bSSFP_gasp_knee_faMax_1x1x2_2D_TR6ms_FID55556.dat',
             'meas_MID124_bSSFP_gasp_knee_faMax_1x1x2_2D_TR12ms_FID55557.dat',
             'meas_MID125_bSSFP_gasp_knee_faMax_1x1x2_2D_TR24ms_FID55558.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset8b(path = os.getcwd()):
    url = 'https://drive.google.com/file/d/1pbn2gsxe-PjvW9vSIE6pnfbCUVWrUGbe/view?usp=sharing'
    dataloader.download_data(url, '20231106_GASP_KNEE', path)
    filepath =  os.path.join(path, 'data', '20231106_GASP_KNEE', '') 
    files = ['meas_MID127_bSSFP_gasp_knee_fa20_1x1x2_2D_TR6ms_FID55560.dat',
             'meas_MID128_bSSFP_gasp_knee_fa20_1x1x2_2D_TR12ms_FID55561.dat',
             'meas_MID129_bSSFP_gasp_knee_fa20_1x1x2_2D_TR24ms_FID55562.dat']

    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset8c(path = os.getcwd()):
    url = 'https://drive.google.com/file/d/1pbn2gsxe-PjvW9vSIE6pnfbCUVWrUGbe/view?usp=sharing'
    dataloader.download_data(url, '20231106_GASP_KNEE', path)
    filepath =  os.path.join(path, 'data', '20231106_GASP_KNEE', '') 
    files = ['meas_MID126_DIXON_2D_3echoes_FID55559.dat']
    M = dataloader.read_rawdata(filepath + files[0])['data']
    return M

def load_dataset9a(path = os.getcwd()):
    ''' Retreives GASP Phantom data for dixon - Experiment from Dec 22, 2023 '''
    url = 'https://drive.google.com/file/d/10ZRAlIO9w5Q3EsJLHXpnIwU14HM9n12L/view?usp=sharing'
    dataloader.download_data(url, '20231222_GASP_PHANTOM', path)
    filepath =  os.path.join(path, 'data', '20231222_GASP_PHANTOM', 'dixon', '') 
    files = os.listdir(filepath)
    print(f'Path: {filepath}')
    print(f'Loading files: {files}')
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset9b(path = os.getcwd()):
    ''' Retreives GASP Phantom data for fa20 - Experiment from Dec 22, 2023 '''
    url = 'https://drive.google.com/file/d/10ZRAlIO9w5Q3EsJLHXpnIwU14HM9n12L/view?usp=sharing'
    dataloader.download_data(url, '20231222_GASP_PHANTOM', path)
    filepath =  os.path.join(path, 'data', '20231222_GASP_PHANTOM', 'gasp_fa20', '') 
    files = os.listdir(filepath)
    print(f'Path: {filepath}')
    print(f'Loading files: {files}')
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset9c(path = os.getcwd()):
    ''' Retreives GASP Phantom data for fa90 - Experiment from Dec 22, 2023 '''
    url = 'https://drive.google.com/file/d/10ZRAlIO9w5Q3EsJLHXpnIwU14HM9n12L/view?usp=sharing'
    dataloader.download_data(url, '20231222_GASP_PHANTOM', path)
    filepath =  os.path.join(path, 'data', '20231222_GASP_PHANTOM', 'gasp_fa90', '') 
    files = os.listdir(filepath)
    print(f'Path: {filepath}')
    print(f'Loading files: {files}')
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset10a(path = os.getcwd()):
    ''' Retreives GASP Ankle data for dixon - Experiment from Dec 22, 2023 '''
    url = 'https://drive.google.com/file/d/10hHegaWbiDYv4MsOt8nXDLpxl1b1xccE/view?usp=sharing'
    dataloader.download_data(url, '20231222_GASP_ANKLE', path)
    filepath =  os.path.join(path, 'data', '20231222_GASP_ANKLE', 'dixon', '') 
    files = os.listdir(filepath)
    print(f'Path: {filepath}')
    print(f'Loading files: {files}')
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset10b(path = os.getcwd()):
    ''' Retreives GASP Ankle data for fa20 - Experiment from Dec 22, 2023 '''
    url = 'https://drive.google.com/file/d/10hHegaWbiDYv4MsOt8nXDLpxl1b1xccE/view?usp=sharing'
    dataloader.download_data(url, '20231222_GASP_ANKLE', path)
    filepath =  os.path.join(path, 'data', '20231222_GASP_ANKLE', 'fa20', '') 
    files = os.listdir(filepath)
    print(f'Path: {filepath}')
    print(f'Loading files: {files}')
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset10c(path = os.getcwd()):
    ''' Retreives GASP Ankle data for fa90 - Experiment from Dec 22, 2023 '''
    url = 'https://drive.google.com/file/d/10hHegaWbiDYv4MsOt8nXDLpxl1b1xccE/view?usp=sharing'
    dataloader.download_data(url, '20231222_GASP_ANKLE', path)
    filepath =  os.path.join(path, 'data', '20231222_GASP_ANKLE', 'fa90', '') 
    files = os.listdir(filepath)
    print(f'Path: {filepath}')
    print(f'Loading files: {files}')
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset11(path = os.getcwd(), foldername='dixon'):
    ''' Retreives GASP Ankle data for fa90 - Experiment from March 12, 2024 '''
    url = 'https://drive.google.com/file/d/1M1WdParsJlWMd5es3ve_e3_lfjS9O-Bl/view?usp=sharing'
    dataloader.download_data(url, '20240312_GASP_PHANTOM', path)
    filepath =  os.path.join(path, 'data', '20240312_GASP_PHANTOM', foldername, '') 
    files = os.listdir(filepath)
    print(f'Path: {filepath}')
    print(f'Loading files: {files}')
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset11a(path = os.getcwd()):
    return load_dataset11(path, 'dixon')

def load_dataset11b(path = os.getcwd()):
    return load_dataset11(path, 'fa20')

def load_dataset11c(path = os.getcwd()):
    return load_dataset11(path, 'fa90')

def load_dataset12(path = os.getcwd(), foldername='dixon'):
    ''' Retreives GASP Phantom for dixon - Experiment from March 27, 2024 '''
    url = 'https://drive.google.com/file/d/1kxqMLtBhsXH0DN4UbgCYwIkjRbSM2CMA/view?usp=sharing'
    dataloader.download_data(url, '20240327_GASP_PHANTOM', path)
    filepath =  os.path.join(path, 'data', '20240327_GASP_PHANTOM', foldername, '')
    files = os.listdir(filepath)
    files.sort()
    print(f'Path: {filepath}')
    print(f'Loading files: {files}')
    M0 = dataloader.read_rawdata(filepath + files[0])['data']
    M1 = dataloader.read_rawdata(filepath + files[1])['data']
    M2 = dataloader.read_rawdata(filepath + files[2])['data']
    M = np.stack([M0,M1,M2], axis=-1)
    return M

def load_dataset12a(path = os.getcwd()):
    return load_dataset12(path, 'dixon')

def load_dataset12b(path = os.getcwd()):
    return load_dataset12(path, 'fa20')

def load_dataset12c(path = os.getcwd()):
    return load_dataset12(path, 'fa90')

def load_dataset12d(path = os.getcwd()):
    return load_dataset12(path, 'dixon2')