"""GASP module."""

import numpy as np
import numpy.typing as npt
from skimage.filters import threshold_li
from scipy.optimize import least_squares

def run_gasp(I: npt.NDArray, An: npt.NDArray, method :str = "affine"):
    ''' Run GASP model on data with shape [Height, Width, PC x TRs] 
    
    Parameters:
    I (NDArray): Array of phase-cycled images.
    D (NDArray): Vector of samples of desired spectral profile.
    method (str, optional): Method used to compute the GASP model solution.
        Must be one of {"affine", "linear", "lev-mar", "lev-mar-quad"}.

    Reuturns:
    NDArray: Reconstructed image.
    '''

    height, width = I.shape[:2]
    I = np.reshape(I, (I.shape[0], I.shape[1], -1))
    I = I.reshape((-1, I.shape[-1]))                
    npcs = I.shape[-1]

    if method == "linear":
        out = I.dot(An).reshape(height, width)
    elif method == "affine":
        I = np.column_stack((np.ones(I.shape[0]), I))
        out = I.dot(An).reshape(height, width)
    elif method == "quad":
        I = np.column_stack((np.ones(I.shape[0]), I, I**2))
        out = I.dot(An).reshape(height, width)
    elif method == "levmar-old":
        x0 = An[:npcs]              # Linear terms
        x1 = An[npcs:]              # Quadratic terms
        out = np.reshape(I @ x0 + I**2 @ x1, (height, width))
    elif method == "levmar-quad":
        c = An[0]                    # Constant term
        x0 = An[1:npcs+1]            # Linear terms
        x1 = An[npcs+1:]             # Quadratic terms
        out = np.reshape(c + I @ x0 + I**2 @ x1, (height, width))
    else:
        raise ValueError(f"method '{method}' was not recognized")

    return out

def train_gasp(I: npt.NDArray, D: npt.NDArray, method: str = "affine"):
    ''' Train GASP model on data with shape [Height, Width, PCs x TRs] and desired spectral profile D with shape [Width,]
    
    Parameters:
    I (NDArray): Array of phase-cycled images.
    D (NDArray): Vector of samples of desired spectral profile.
    method (str, optional): Method used to compute the GASP model solution.
        Must be one of {"affine", "linear", "lev-mar", "lev-mar-quad"}.

    Reuturns:
    tuple: Reconstructed image and coefficients.
    '''

    # Reshape the data to be in the form [Height x Width, PCs x TRs]
    height, width = I.shape[:2]
    I = I.reshape((-1, I.shape[-1]))    # Collapse all dimensions last dimension
    npcs = I.shape[-1]

    # Repeat the desired spectral profile to match the number of PCs
    D = np.tile(D, (int(I.shape[0]/D.size),))

    # Now solve the system
    if method == "linear":
        A = np.linalg.lstsq(I, D, rcond=None)[0]  # Solves a linear system of form: D = A * I (i.e. y = a * x)
        out = I.dot(A).reshape(height, width)     # Reconstruct the image from the coefficients 
    elif method == "affine":
        I = np.column_stack((np.ones(I.shape[0]), I))  # Add a column of ones (b) to the data so data is from of y = a * x + b
        A = np.linalg.lstsq(I, D, rcond=None)[0]  # Solves a linear system of form: D = A * I (i.e. y = a * x)
        out = I.dot(A).reshape(height, width)     # Reconstruct the image from the coefficients 
    elif method == "quad":
        I_quad = np.column_stack((np.ones(I.shape[0]), I, I**2))  # Add columns for constant, linear, and quadratic terms
        A = np.linalg.lstsq(I_quad, D, rcond=None)[0]  # Solves a quadratic system of form: D = a * I^2 + b * I + c
        out = I_quad.dot(A).reshape(height, width) 
    elif method == 'levmar-old':
        def residuals(y):
            y0 = y[:npcs] + 1j*y[npcs:2*npcs]
            y1 = y[2*npcs:3*npcs] + 1j*y[3*npcs:4*npcs]
            residual = I @ y0 + I**2 @ y1 - D
            return np.concatenate((residual.real, residual.imag))

        res = least_squares(fun=residuals, x0=np.zeros(npcs*4), method="lm")
        if not res.success:
            print(f"GASP SOLVE ERROR ({method}): {res.message}")

        x0 = res.x[:npcs] + 1j*res.x[npcs:2*npcs]
        x1 = res.x[2*npcs:3*npcs] + 1j*res.x[3*npcs:4*npcs]
        A = np.concatenate((x0, x1))
        out = np.reshape(I @ x0 + I**2 @ x1, (height, width))
    elif method == 'levmar-quad':
        def quadratic_model_residuals(params):
            c = params[0] + 1j*params[1]                                    # Constant term
            y0 = params[2:npcs+2] + 1j*params[npcs+2:2*npcs+2]              # Linear terms
            y1 = params[2*npcs+2:3*npcs+2] + 1j*params[3*npcs+2:4*npcs+2]   # Quadratic terms
            residual = c + I @ y0 + I**2 @ y1 - D
            return np.concatenate((residual.real, residual.imag))

        # Initialize parameters
        initial_params = np.zeros(4*npcs + 2)
        initial_params[0] = np.real(np.mean(D))  # Real part of constant
        initial_params[1] = np.imag(np.mean(D))  # Imaginary part of constant

        res = least_squares(fun=quadratic_model_residuals, x0=initial_params, method="lm")
        if not res.success:
            print(f"GASP SOLVE ERROR ({method}): {res.message}")
        
        c = res.x[0] + 1j*res.x[1]
        x0 = res.x[2:npcs+2] + 1j*res.x[npcs+2:2*npcs+2]
        x1 = res.x[2*npcs+2:3*npcs+2] + 1j*res.x[3*npcs+2:4*npcs+2]
        A = np.concatenate(([c], x0, x1))
        out = np.reshape(c + I @ x0 + I**2 @ x1, (height, width))
    else:
        raise ValueError(f"method '{method}' was not recognized")
    
    A = np.array(A)
    return out, A

def create_data_mask(M):
    # Create mask of phantom
    _ = np.sqrt(np.sum(np.abs(M)**2, axis=2))
    _ = np.mean(_, axis=2)
    _ = np.mean(_, axis=2)
    _ = abs(_)
    thresh = threshold_li(_)
    mask = np.abs(_) > thresh
    return mask

def process_data_for_gasp(M, D=None, useMask=False, useCalibration=False, clines=2):

    height = M.shape[0]
    width = M.shape[1]

    # Use mask to remove background from data for training
    if useMask:
        # Create mask of phantom
        _ = np.sqrt(np.sum(np.abs(M)**2, axis=2))
        _ = np.mean(_, axis=2)
        _ = np.mean(_, axis=2)
        _ = abs(_)
        thresh = threshold_li(_)
        mask = np.abs(_) > thresh

        # Apply mask to data
        mask0 = np.tile(
            mask, (M.shape[2:] + (1, 1,))).transpose((3, 4, 0, 1, 2))
        data = M * mask0
    else:
        data = M

    if useCalibration:
        # Extract calibration region
        C_dim = (clines, width) # Calibration box - (# Number of lines of calibration, Pixels on signal)
        mid = [d // 2 for d in data.shape[:2]]
        pad = [d // 2 for d in C_dim]
        data = data[mid[0]-pad[0]:mid[0]+pad[0], mid[1]-pad[1]:mid[1]+pad[1], :]

        if D is not None:
            D = D[mid[1]-pad[1]:mid[1]+pad[1]]

    # Reshape data to required shapes [Height, Width, Coil, PCs, TRs] -> [Coil, Height, Width, PCs x TRs]
    data = np.reshape(data, data.shape[:-2] + (-1,))    # [Height, Width, Coil, PCs x TRs] - Combine coils and TRs
    data = np.moveaxis(data, 2, 0)                      # [Coil, Height, Width, PCs x TRs] - Move coils to first axis

    return data

def train_gasp_with_coils(data, D, method="linear"):

    # Get new dimensions
    ncoils, height, width, npcs = data.shape[:]

    # Run gasp
    Ic = np.zeros((ncoils, height, width), dtype='complex')
    An = np.zeros((ncoils, npcs+1))
    for cc in range(ncoils):
        Ic[cc, ...], An[cc, ...] = train_gasp(data[cc, ...], D, method=method)
    Ic = np.sqrt(np.sum(np.abs(Ic)**2, axis=0))

    return Ic, An

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


def apply_gasp(I, An, method:str = "linear"):
    ''' Use gasp model on input magenatization data, I. Shape should be [height, width, PC x TRs]'''
    #I = np.squeeze(I).transpose(1, 2, 0)
    xx, yy = I.shape[0], I.shape[1]
    
    if method == "linear":
        out = I.dot(An).reshape(xx, yy)
    elif method == "lev-mar":
        out = np.reshape(I @ An, (xx, yy))
    elif method == "lev-mar-quad":
        x0 = An[:An.shape[0] // 2, ...]
        x1 = An[An.shape[0] // 2:, ...]
        out = np.reshape(I @ x0 + I**2 @ x1, (xx, yy))
    else:
        raise ValueError(f"method '{method}' was not recognized")

    return out
