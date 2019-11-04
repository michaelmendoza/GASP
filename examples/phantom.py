
import os
import sys
sys.path.insert(0, './')

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mr_utils import view
from mr_utils.recon.ssfp import gs_recon
from tqdm import trange
from skimage.filters import threshold_li
from time import time

from gasp import gasp, triangle, triangle_periodic, data_preprocessing

def examine_phantom(path, TEs=[12, 24, 48]):

    if not os.path.exists(path + '/gasp_data.npy'):
        data = data_preprocessing(path)
    else:
        data = np.load(path + '/gasp_data.npy')

    print('Starting Processing ...')
    print('Data shape: ' + str(data.shape))
    
def water_phantom(path, TEs=[12, 24, 48]):

    if not os.path.exists(path + '/gasp_data.npy'):
        data = data_preprocessing(path)
    else:
        data = np.load(path + '/gasp_data.npy')

    print('Starting Processing ...')
    print(data.shape)

    # Create a mask of the phanton
    band_free = gs_recon(data[:, :, 0, ::4, 0], pc_axis=-1)
    thresh = threshold_li(np.abs(band_free))
    mask = np.abs(band_free) > thresh

    # Apply mask to data
    mask0 = np.tile(mask, (data.shape[2:] + (1, 1,))).transpose((3, 4, 0, 1, 2))
    data = data * mask0
    
    # Reshape data   
    data = np.reshape(data, data.shape[:-2] + (-1,))    # [Height, Width, Coil, PCs x TRs]
    data = np.moveaxis(data, 2, 0)                      # [Coil, Height, Width, PCs x TRs]
    data = data.transpose((0, 3, 2, 1))                 # [Coil, PCs x TRs, Width, Height]

    # Get new dimensions
    ncoils, npcs, height, width = data.shape[:]

    # Calibration box - (# Number of lines of calibration, Pixels on signal)
    C_dim = (32, width)
    D = triangle_periodic(width, 76, 18, 38)
    D *= mask.T[int(height/2), :]

    plt.plot(D)
    plt.show()

    Ic = np.zeros((ncoils, height, width), dtype='complex')
    for cc in trange(ncoils, leave=False):
        Ic[cc, ...] = gasp(data[cc, ...], D, C_dim, pc_dim=0)
    Ic = np.sqrt(np.sum(np.abs(Ic)**2, axis=0))
    #Ic = np.abs(Ic[2, ...])

    plt.subplot(1, 3, 1)
    plt.imshow(np.sqrt(np.sum(abs(data[:, 0, ...])**2, axis=0)))

    plt.subplot(1, 3, 2)
    plt.imshow(Ic)

    plt.subplot(1, 3, 3)
    plt.plot(np.abs(Ic[int(height/2), :]), label='Simulated Profile')
    plt.plot(D, '--', label='Desired Profile')
    plt.legend()
    plt.title('Center horizontal slice\'s spatial response profile')
    plt.show()

if __name__ == '__main__':
    water_phantom('/Volumes/NO NAME/Data/GASP/20190507_GASP_LONG_TR_WATER')


