"""Reproduce bandpass filter in simulation."""

import matplotlib.pyplot as plt
import numpy as np
from phantominator import shepp_logan
from ssfp import bssfp

from gasp.gasp import gasp


if __name__ == "__main__":
    L, M, N = 64, 65, 1
    M0, T1, T2 = shepp_logan(N=(L, M, N), MR=True, zlims=(-.25, .25))
    M0 = M0.squeeze()
    T1 = T1.squeeze()
    T2 = T2.squeeze()

    #PCs = (0, np.deg2rad(90))
    PCs = np.linspace(0, 4*np.pi, 8)
    nPCs = len(PCs)
    TRs = (3e-3, 6e-3, 12e-3, 24e-3)
    #TRs = (24e-3,)
    nTRs = len(TRs)

    # Simple linear gradient off-resonance
    df_factor = 4
    maxTR = np.max(TRs)
    minTR = np.min(TRs)
    fx = np.linspace(-df_factor / maxTR, df_factor / maxTR, M)
    fy = np.zeros(L)
    df, _ = np.meshgrid(fx, fy)

    # simulate acquisitions
    sims = np.empty((nTRs, nPCs, L, M), dtype=np.complex128)  # (TRs, PCs, x, y)
    for ii, TR in enumerate(TRs):
        for jj, PC in enumerate(PCs):
            sims[ii, jj, ...] = bssfp(T1=T1, T2=T2, TR=TR, alpha=np.deg2rad(90), field_map=df, phase_cyc=PC, M0=M0)

    # collapse TR/PC dims to single spectral dim
    sims = np.reshape(sims, (nTRs*nPCs, L, M))

    # normalize
    sims /= np.abs(sims.flatten()).max()

    # plt.imshow(np.abs(sims[0, ...]))
    # plt.show()

    # # show some voxel's spectral profile
    # pt = (L//2, M//2)
    # sig = sims[:, pt[0], pt[1]]
    # fig, ax1 = plt.subplots()
    # pc_labels = []
    # for TR in TRs:
    #     pc_labels += [f"TR {TR} \n {PC} deg" for PC in np.rad2deg(PCs)]
    # ax1.plot(pc_labels, np.abs(sig), 'k-')
    # ax1.set_ylabel('Magnitude (a.u.)')
    # ax1.set_xlabel('Phase Cycle (in deg)')
    # ax2 = ax1.twinx()
    # ax2.plot(pc_labels, np.angle(sig), 'k--')
    # ax2.set_ylabel('Phase (rad)')
    # plt.show()

    # desired spatial profile (?)
    bandpass = np.zeros(sims.shape[1:])
    bandpass[:, 3*M//8:5*M//8] = 1
    plt.imshow(bandpass)
    plt.show()
    bandpass = np.reshape(bandpass, (-1,))
    # bandpass -= 1
    # bandpass = np.abs(bandpass)

    # x = np.linalg.lstsq(np.reshape(sims, (sims.shape[0], -1)).T, bandpass, rcond=None)[0]
    # res = (x @ sims.reshape(x.size, -1)).reshape(sims.shape[1:])
    C_dim = (L, M)
    res = gasp(sims, bandpass, C_dim=C_dim, pc_dim=0)

    plt.imshow(np.abs(res))
    plt.show()
