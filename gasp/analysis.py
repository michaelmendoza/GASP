import numpy as np
import matplotlib.pyplot as plt

def dixon_3pt(M):
    ''' 3pt dixon - M data with shape [height, width, coils, TEs]'''
    M0 = M[:,:,:,0]
    M1 = M[:,:,:,1]
    M2 = M[:,:,:,2]

    field_map = np.angle(np.conjugate(M0) * M2)
    f = np.exp(-1j * field_map / 2)
    Mw = M0 + M1
    Mf = M0 - M1
    Mw3 = M0 + M1 * np.exp(-1j * field_map / 2)
    Mf3 = M0 - M1 * np.exp(-1j * field_map / 2)

    # 2-pt dixon
    Mw = np.sqrt(np.sum(np.abs(Mw)**2, axis=2))
    Mf = np.sqrt(np.sum(np.abs(Mf)**2, axis=2))

    # 3-pt dixon
    Mw3 = np.sqrt(np.sum(np.abs(Mw3)**2, axis=2))
    Mf3 = np.sqrt(np.sum(np.abs(Mf3)**2, axis=2))

    return Mw3, Mf3

def plot_dixon(Mw, Mf):
    f = plt.figure(figsize=(8,8))
    ax = f.add_subplot(2, 2, 1)
    ax2 = f.add_subplot(2, 2, 2)
    ax.imshow(np.abs(Mw), cmap='gray')
    ax2.imshow(np.abs(Mf), cmap='gray')
    plt.plot()