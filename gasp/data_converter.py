from glob import glob
from os.path import isfile

import numpy as np
from rawdatarinator.raw import raw

def convert_to_npy(foldername):
    
  files = glob(foldername + '/meas_*.dat')

  for f in files:
      new_filename = '%s.npy' % f
      if not isfile(new_filename):
          data = raw(f)['kSpace']
          data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(
              data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
          np.save(new_filename, data)
          print('Done with %s' % f)