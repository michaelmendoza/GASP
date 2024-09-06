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

    # Reshape the data to be in the form [Height x Width, PCs x TRs]
    height, width = I.shape[:2]
    I = I.reshape(I.shape[0], I.shape[1], -1)   # Collapse all but first 2 dimensions
    I = I.reshape((-1, I.shape[-1]))            # Collapse all dimensions last dimension
    npcs = I.shape[-1]
    data_shape = (height, width)

    if method == "linear":
        out = I.dot(An).reshape(data_shape)
    elif method == "affine":
        I = np.column_stack((np.ones(I.shape[0]), I))
        out = I.dot(An).reshape(data_shape)
    elif method == "quad":
        I = np.column_stack((np.ones(I.shape[0]), I, I**2))
        out = I.dot(An).reshape(data_shape)
    elif method == "levmar-old":
        x0 = An[:npcs]              # Linear terms
        x1 = An[npcs:]              # Quadratic terms
        out = np.reshape(I @ x0 + I**2 @ x1, data_shape)
    elif method == "levmar-quad":
        c = An[0]                    # Constant term
        x0 = An[1:npcs+1]            # Linear terms
        x1 = An[npcs+1:]             # Quadratic terms
        out = np.reshape(c + I @ x0 + I**2 @ x1, data_shape)
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
    I = I.reshape(I.shape[0], I.shape[1], -1)   # Collapse all but first 2 dimensions
    I = I.reshape((-1, I.shape[-1]))            # Collapse all dimensions last dimension
    npcs = I.shape[-1]
    data_shape = (height, width)

    # Repeat the desired spectral profile to match the number of PCs
    D = np.tile(D, (int(I.shape[0]/D.size),))

    # Now solve the system
    if method == "linear":
        A = np.linalg.lstsq(I, D, rcond=None)[0]  # Solves a linear system of form: D = A * I (i.e. y = a * x)
        out = I.dot(A).reshape(data_shape)     # Reconstruct the image from the coefficients 
    elif method == "affine":
        I = np.column_stack((np.ones(I.shape[0]), I))  # Add a column of ones (b) to the data so data is from of y = a * x + b
        A = np.linalg.lstsq(I, D, rcond=None)[0]  # Solves a linear system of form: D = A * I (i.e. y = a * x)
        out = I.dot(A).reshape(data_shape)     # Reconstruct the image from the coefficients 
    elif method == "quad":
        I_quad = np.column_stack((np.ones(I.shape[0]), I, I**2))  # Add columns for constant, linear, and quadratic terms
        A = np.linalg.lstsq(I_quad, D, rcond=None)[0]  # Solves a quadratic system of form: D = a * I^2 + b * I + c
        out = I_quad.dot(A).reshape(data_shape) 
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
        out = np.reshape(I @ x0 + I**2 @ x1, data_shape)
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
        out = np.reshape(c + I @ x0 + I**2 @ x1, data_shape)
    else:
        raise ValueError(f"method '{method}' was not recognized")
    
    A = np.array(A)
    return out, A

def train_gasp_with_coils(data, D, method="affine"):

    # Get dimensions
    if data.ndim == 4:
        height, width, ncoils, npcs = data.shape[:]
    if data.ndim == 5:
        height, width, ncoils, npcs, TRs = data.shape[:]
        npcs = npcs * TRs       # Combine PCs and TRs

    # Run gasp
    out = np.zeros((ncoils, height, width), dtype='complex')
    n = npcs+1 if method == 'affine' else npcs
    An = np.zeros((ncoils, n), dtype='complex')
    for cc in range(ncoils):
        single_coil = data[:,:, cc, ...].reshape(height, width, npcs)
        out[cc, ...], An[cc, ...] = train_gasp(single_coil, D, method=method)
    out = np.sqrt(np.sum(np.abs(out)**2, axis=0))
    
    return out, An

def create_data_mask(M):
    # Create mask of phantom
    _ = np.sqrt(np.sum(np.abs(M)**2, axis=2))
    _ = np.mean(_, axis=2)
    _ = np.mean(_, axis=2)
    _ = abs(_)
    thresh = threshold_li(_)
    mask = np.abs(_) > thresh
    return mask

def apply_mask_to_data(M, mask):
    # Ensure mask has the same shape as the first two dimensions of M
    if mask.shape != M.shape[:2]:
        raise ValueError("Mask shape does not match the first two dimensions of the data")

    if len(M.shape) == 5: 
        # Expand mask to match the dimensions of M using np.tile
        expanded_mask = mask[:, :, np.newaxis, np.newaxis, np.newaxis]
        expanded_mask = np.tile(expanded_mask, (1, 1, M.shape[2], M.shape[3], M.shape[4]))

    # Apply the mask
    masked_data = M * expanded_mask

    return masked_data

def extract_centered_subset(data, n_lines):
    """
    Extract a subset of the data centered around the middle of the height dimension.
    
    Args:
    data (np.ndarray): Input data with shape [height, width, coils, pcs, TR]
    n_lines (int): Number of lines to extract in the height dimension
    
    Returns:
    np.ndarray: Subset of the data with shape [n_lines, width, coils, pcs, TR]
    """
    if n_lines > data.shape[0]:
        raise ValueError("n_lines cannot be greater than the height of the data")
    
    # Calculate the start and end indices for the subset
    center = data.shape[0] // 2
    start = center - n_lines // 2
    end = start + n_lines
    
    # Extract the subset
    subset = data[start:end, ...]
    
    return subset

def process_data_for_gasp(M, useMask=False, useCalibration=False, n_lines=2):
    ''' Process data for GASP model training.
    
    Parameters:
    M (NDArray): Array of phase-cycled images.
    D (NDArray): Vector of samples of desired spectral profile.
    useMask (bool, optional): Whether to use a mask to remove background from the data.
    useCalibration (bool, optional): Whether to use a calibration region.
    n_lines (int, optional): Number of lines to extract from the center of the data.
    
    Returns:
    NDArray: Processed data.
    '''

    if useMask:
        mask = create_data_mask(M)
        data = apply_mask_to_data(M, mask)
    else:
        data = M

    if useCalibration:
        data = extract_centered_subset(data, n_lines)

    return data
