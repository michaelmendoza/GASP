'''Convert raw data to npy for use with this software package.'''

from glob import glob
from os.path import isfile

import numpy as np
from rawdatarinator.raw import raw

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
            data = raw(f)['kSpace']
            data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(
                data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
            np.save(new_filename, data)
            print('Done with %s' % f)
