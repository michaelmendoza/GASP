'''Convert raw data to npy for use with this software package.'''

from glob import glob
from os.path import isfile
from time import time

import numpy as np
from rawdatarinator import twixread

def convert_to_npy(foldername, regex='/meas_*.dat'):
    '''Convert all raw data files in a folder.
    Parameters
    ----------
    foldername : str
        Path to directory where the raw data files are stored.
    regex : str, optional
        Regular expression to match in the directory.
    Returns
    -------
    None
    '''

    files = glob(foldername + regex)

    for f in files:
        new_filename = '%s.npy' % f
        if not isfile(new_filename):
            # data = raw(f)['kSpace']
            data = twixread(f, A=True).squeeze()
            data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(
                data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
            data = _data.transpose((0, 1, 2, 4, 3))  
            np.save(new_filename, data)
            print('Done with %s' % f)

def data_preprocessing(path, TEs = [12, 24, 48], regex='/meas_*.dat'):
    '''Converts raw data to preprocessed data file. 

    This loads module loads data, computes iFFT, stacks data, and removes 
    averages.
    '''

    print('PreProcessing Starting: ')
    files = glob(path + regex)

    # Load data, compute iFFT, and stack data 
    t0 = time()
    data = []
    for ii, te in enumerate(TEs):
        f = files[ii]

        print('Loading and PreProcessing: TE = ' + str(te) + ' Path:' + f)

        _data = twixread(f, A=True).squeeze()
        _data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(
                _data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        _data = _data.transpose((0, 1, 2, 4, 3)) 
        data.append(_data[..., None])
        
    data = np.concatenate(data, axis=-1)
    print('Data loaded in ' + str(time() - t0) + ' secs')
    print(data.shape) # [Height, Width, Coil, Avg, PCs, TRs]
    
    # Collapse the averages dimension
    print('PreProcessing: Averaging')
    data = np.mean(data, axis=3) # [Height, Width, Coil, PCs, TRs]
    print(data.shape) 

    # Save PreProcessed data to Savefile 
    new_filename = path + '/gasp_data.npy';
    print('Saving PreProcessed Data file')
    np.save(new_filename, data)
    print('PreProcessing Complete with %s' % new_filename)

    return data

if __name__ == '__main__':
  # convert_to_npy("/Volumes/NO NAME/Data/GASP/20190507_GASP_LONG_TR_WATER")
  # average_and_concate_data("/Volumes/NO NAME/Data/GASP/20190507_GASP_LONG_TR_WATER")

  # Test new rawdatarinator
  # convert_to_npy('data/20190401_GASP_PHANTOM/')
  # data = np.load('data/20190401_GASP_PHANTOM/meas_MID48_TRUFI_NBPM_2019_02_27_FID41503.dat.npy')
  # print(data.shape)
  pass