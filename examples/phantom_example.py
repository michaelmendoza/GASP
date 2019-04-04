
import sys
sys.path.insert(0, './')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mr_utils import view
from tqdm import trange

from gasp import gasp, triangle

if __name__ == '__main__':
  
  dataset = 2
  data0 = np.load('data/20190401_GASP_PHANTOM/set%d_tr6_te3.npy' % dataset)
  data1 = np.load('data/20190401_GASP_PHANTOM/set%d_tr12_te6.npy' % dataset)
  data2 = np.load('data/20190401_GASP_PHANTOM/set%d_tr24_te12.npy' % dataset)

  print(data0.shape) # [Height, Width, Coil, Avg, PCs]

  data = np.concatenate((data0[..., None], data1[..., None], data2[..., None]), axis=-1)
  data = np.mean(data, axis=3) # [Height, Width, Coil, PCs, TRs]
  print(data.shape[:-2])
  data = np.reshape(data, data.shape[:-2] + (-1,)) # [Height, Width, Coil, PCs x TRs]
  data = np.moveaxis(data, 2, 0) # [Coil, Height, Width, PCs x TRs]
  data = data.transpose((0, 3, 2, 1)) # [Coil,  PCs x TRs, Width, Height]
  ncoils, npcs, width, height = data.shape[:]

  # view(data, fft_axes=(-2, -1), montage_axis=0, movie_axis=1, movie_interval=200)

  D = triangle(np.linspace(-100, 100, height), bw=10)
  D = np.roll(D, int(height/9))
  plt.plot(D)
  plt.show()

  Ic = np.zeros((ncoils, width, height), dtype='complex')
  for cc in trange(ncoils, leave=False):
    Ic[cc, ...] = gasp(data[cc, ...], D, pc_dim=0)
  Ic = np.sqrt(np.sum(np.abs(Ic)**2, axis=0))

  plt.subplot(1, 3, 1)
  plt.imshow(np.sqrt(np.sum(abs(data[:, 0, ...])**2, axis=0)))

  plt.subplot(1, 3, 2)
  plt.imshow(Ic)

  plt.subplot(1, 3, 3)
  plt.plot(np.abs(Ic[int(width/2), :]), label='Simulated Profile')
  plt.plot(D, '--', label='Desired Profile')
  plt.legend()
  plt.title('Center horizontal slice\'s spatial response profile')
  plt.show()