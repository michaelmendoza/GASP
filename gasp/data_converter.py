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
            np.save(new_filename, data)
            print('Done with %s' % f)


def average_and_concate_data(path):
    new_filename = path + '/gasp_data.npy'
    print(new_filename)
    dataset = 1
    TEs = [12, 24, 48]

    data = []
    t0 = time()
    print('Loading Data ...')
    for ii, te in enumerate(TEs):
        data.append(np.load('%s/set%d_tr%d_te%d.npy' % (
            path, dataset, te*2, te))[..., None])
    data = np.concatenate(data, axis=-1)

    print('Data loaded in ' + str(time() - t0) + ' secs')
    print(data.shape) # [Height, Width, Coil, Avg, PCs, TRs]

    # Collapse the averages dimension
    data = np.mean(data, axis=3) # [Height, Width, Coil, PCs, TRs]
    print(data.shape)

    np.save(new_filename, data)
    print('Done with %s' % new_filename)

if __name__ == '__main__':
  #convert_to_npy("/Volumes/NO NAME/Data/GASP/20190507_GASP_LONG_TR_WATER")
  # average_and_concate_data("/Volumes/NO NAME/Data/GASP/20190507_GASP_LONG_TR_WATER")

  # Test new rawdatarinator
  # convert_to_npy('data/20190401_GASP_PHANTOM/')
  # data = np.load('data/20190401_GASP_PHANTOM/meas_MID48_TRUFI_NBPM_2019_02_27_FID41503.dat.npy')
  # print(data.shape)
