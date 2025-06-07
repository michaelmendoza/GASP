import numpy as np
import matplotlib.pyplot as plt

def coil_combine(data, type='mag'):
    if type == 'mag':
        return coil_combine_mag(data)

def coil_combine_mag(data):
    """
    Combine multi-coil MRI data by magnitude-weighted complex sum.
    
    Parameters
    ----------
    data : np.ndarray (complex)
        The MRI data of shape [Nx, Ny, Ncoils, Npc, NTR].
    
    Returns
    -------
    combined : np.ndarray (complex)
        Coil-combined data of shape [Nx, Ny, Npc, NTR].
    """
    eps=1e-8

    # Magnitude of each coil
    weights = np.abs(data)  # shape [Nx, Ny, Ncoils, NphaseCycles, NTR]
    
    # Sum over coil dimension with the coil's magnitude as weight
    numerator = np.sum(weights * data, axis=2)         # shape [Nx, Ny, NphaseCycles, NTR]
    denominator = np.sum(weights, axis=2) + eps        # same shape, add eps to avoid div by zero
    
    combined = numerator / denominator
    
    #print(combined.shape)
    #plt.imshow(np.abs(combined[:,:,0,0]), cmap='gray')

    return combined