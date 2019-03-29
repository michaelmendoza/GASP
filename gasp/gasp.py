import numpy as np
import matplotlib.pyplot as plt

def gasp(I, D):
    '''Generation of Arbitrary Spectral Profiles.

    Parameters
    ----------
    I : array_like
        Vector of phase-cycles, samples of implicit spectral profile.
    D : array_like
        Vector of samples of desired spectral profile.

    Returns
    -------
    I0 : array_like
        Closest approximation to D.
    '''

    # Ix = D
    # x = np.linalg.pinv(I[None, :]).dot(D[None, :])
    x = np.linalg.pinv(I[:, None]).dot(D[:, None])
    print(x)
    return I.dot(x)

if __name__ == '__main__':
    pass
