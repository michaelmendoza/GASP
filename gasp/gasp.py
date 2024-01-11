"""GASP module."""

import numpy as np


def gasp(I, D, C_dim, pc_dim: int = 0, method: str = "linear"):
    """Generation of Arbitrary Spectral Profiles.

    Parameters
    ----------
    I : array_like
        Array of phase-cycled images.
    D : array_like
        Vector of samples of desired spectral profile.
    C_dim: tuple
        Calibration box dimensions in number of pixels.
    pc_dim : int, optional
        Axis containing phase-cycles.
    method : str, optional
        Method used to compute the least-squares solution.
        Must be one of {"linear", "lev-mar", "lev-mar-quad"}.

    Returns
    -------
    I0 : array_like
        Combined image with spatial response approximating D.
    """

    out, An = gasp_coefficients(I=I, D=D, C_dim=C_dim, pc_dim=pc_dim, method=method)

    return out


def gasp_coefficients(I, D, C_dim, pc_dim: int=0, method: str = "linear"):
    """Generation of Arbitrary Spectral Profiles.

    Parameters
    ----------
    I : array_like
        Array of phase-cycled images.
    D : array_like
        Vector of samples of desired spectral profile.
    C_dim: tuple
        Calibration box dimensions in number of pixels.
    pc_dim : int, optional
        Axis containing phase-cycles.
    method : str, optional
        Method used to compute the least-squares solution.
        Must be one of {"linear", "lev-mar", "lev-mar-quad"}.

    Returns
    -------
    I0 : array_like
        Combined image with spatial response approximating D.
    A0 : array_like
        GASP coefficients.
    """

    # Let's put the phase-cycle dimension last
    I = np.moveaxis(I, pc_dim, -1)
    I0 = I.copy()

    # Save the in-plane dimensions for reshape at end
    xx, yy = I.shape[:2]

    mid = [int(xx/2), int(yy/2)]
    pad = [int(C_dim[0]/2), int(C_dim[1]/2)]
    I = I[mid[0]-pad[0]:mid[0]+pad[0], mid[1]-pad[1]:mid[1]+pad[1], :]
    D = D[mid[1]-pad[1]:mid[1]+pad[1]]

    # import matplotlib.pyplot as plt
    # plt.imshow(np.abs(I[:, :, 0]))
    # plt.show()
    # print(I.shape)

    #view(I, movie_axis=-1)

    # Now let's put all the voxels' time curves down the first dim
    I = I.reshape((-1, I.shape[-1]))

    # Now repeat the desired spectral profile the correct number of
    # times to line up with the length of each column
    D = np.tile(D, (int(I.shape[0]/D.size),))

    # Now solve the system
    if method == "linear":
        x = np.linalg.lstsq(I, D, rcond=None)[0]
        out = I0.dot(x).reshape(xx, yy)
    elif method == "lev-mar":
        from scipy.optimize import least_squares
        npcs = I.shape[-1]

        def _fun(y):
            """
            I @ y = D
            => residual f(x) = I @ y - D
            """
            y0 = y[:npcs] + 1j*y[npcs:2*npcs]
            residual = I @ y0 - D
            return np.concatenate((residual.real, residual.imag))

        res = least_squares(fun=_fun, x0=np.zeros(npcs*2), method="lm")
        if not res.success:
            print(f"GASP SOLVE ERROR ({method}): {res.message}")
        x = res.x[:npcs] + 1j*res.x[npcs:2*npcs]
        out = np.reshape(I0 @ x, (xx, yy))
    elif method == "lev-mar-quad":
        from scipy.optimize import least_squares
        npcs = I.shape[-1]

        def _fun(y):
            """
            I @ y = D
            => residual f(x) = I @ y - D
            """
            y0 = y[:npcs] + 1j*y[npcs:2*npcs]
            y1 = y[2*npcs:3*npcs] + 1j*y[3*npcs:4*npcs]
            residual = I @ y0 + I**2 @ y1 - D
            return np.concatenate((residual.real, residual.imag))

        res = least_squares(fun=_fun, x0=np.zeros(npcs*4), method="lm")
        if not res.success:
            print(f"GASP SOLVE ERROR ({method}): {res.message}")
        x0 = res.x[:npcs] + 1j*res.x[npcs:2*npcs]
        x1 = res.x[2*npcs:3*npcs] + 1j*res.x[3*npcs:4*npcs]
        x = np.concatenate((x0, x1))
        out = np.reshape(I0 @ x0 + I0**2 @ x1, (xx, yy))
    else:
        raise ValueError(f"method must be one of {{'linear', 'lev-mar'}}; got '{method}' instead")

    return out, x

def apply_gasp(I, An):
    ''' Use gasp model on input magenatization data, I. Shape should be [height, width, PC x TRs]'''
    #I = np.squeeze(I).transpose(1, 2, 0)
    xx, yy = I.shape[0], I.shape[1]
    return I.dot(An).reshape(xx, yy)
