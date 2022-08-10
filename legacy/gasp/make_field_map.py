import numpy as np
from ismrmrdtools.coils import (
    calculate_csm_inati_iter, calculate_csm_walsh)

from mr_utils import view
from mr_utils.recon.field_map import dual_echo_gre

if __name__ == '__main__':

  im0 = np.load("/Volumes/NO NAME/Data/GASP/20190401_GASP_PHANTOM/meas_MID57_gre_TR34_TE2_87_FID41512.dat.npy")
  im0 = np.mean(im0, axis=2)
  im0 = np.moveaxis(im0, -1, 0)

  im1 = np.load("/Volumes/NO NAME/Data/GASP/20190401_GASP_PHANTOM/meas_MID58_gre_TR34_TE5_74_FID41513.dat.npy")
  im1 = np.mean(im1, axis=2)
  im1 = np.moveaxis(im1, -1, 0)

  # Make a field map coil by coil
  fm0 = dual_echo_gre(im0, im1, 2.87e-3, 5.74e-3)
  np.save("/Volumes/NO NAME/Data/GASP/20190401_GASP_PHANTOM/coil_fm_gre.npy", fm0)
  view(fm0)
  fm0 = np.mean(fm0, axis=0)

  # Coil combine im0 and im1 then get field map
  _, im0cc0 = calculate_csm_inati_iter(im0)
  _, im1cc0 = calculate_csm_inati_iter(im1)
  csm, _ = calculate_csm_walsh(im0)
  im0cc1 = np.sum(np.conj(im0)*csm, axis=0)
  csm, _ = calculate_csm_walsh(im1)
  im1cc1 = np.sum(np.conj(im1)*csm, axis=0)
  fm1 = dual_echo_gre(im0cc0, im1cc0, 2.87e-3, 5.74e-3)
  fm2 = dual_echo_gre(im0cc1, im1cc1, 2.87e-3, 5.74e-3)

  # Compare
  view(np.stack((fm0, fm1, fm2)))
