
import sys
sys.path.insert(0, './')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mr_utils import view
from mr_utils.recon.ssfp import gs_recon
from tqdm import trange
from skimage.filters import threshold_li
from time import time

from gasp import gasp, triangle, triangle_periodic 

if __name__ == '__main__':
  
  dataset = 2
  t0 = time()
  data0 = np.load('data/20190401_GASP_PHANTOM/set%d_tr6_te3.npy' % dataset)
  #data1 = np.load('data/20190401_GASP_PHANTOM/set%d_tr12_te6.npy' % dataset)
  #data2 = np.load('data/20190401_GASP_PHANTOM/set%d_tr24_te12.npy' % dataset)

  print(data0.shape) # [Height, Width, Coil, Avg, PCs]

  height, width, coil, avg, pcs = data0.shape
  I = data0[int(height/2), int(width/2), 0, 0, :]

  plt.subplot(2,1,1)
  plt.plot(np.abs(I))
  plt.subplot(2,1,2)
  plt.plot(np.angle(I))

  plt.show()