'''Simple cylindrical numerical phantom.'''

import numpy as np

def get_cylinder(N, df_range=None, radius=None, PD=1, T1=1.2, T2=.035):
    '''Axial slice of cylindrical phantom.

    Parameters
    ----------
    N : int
        Matrix height and width.
    df_range : tuple, optional
        Min and max values for linear off-resonance map (in Hz).
    radius : float, optional
        Normalized radius of cylinder (0 < radius < 1).
    PD : float, optional
        Proton Density
    T1 : float, optional
        T1 value in seconds
    T2 : float, optional
        T2 value in seconds
    
    Returns
    -------
    PDs : array_like
        Matrix of proton density values (arbitrary units).
    T1s : array_like
        Matrix of T1 values (in sec).
    T2s : array_like
        Matrix of T2 values (in sec).
    df : array_like
        Matrix of off-resonance values (in Hz).
    '''

    # If we don't have a radius, get default radius
    if radius is None:
        radius = 0.75
    else:
        assert 0 < radius < 1, 'Radius must be in (0, 1)!'


    # Find indices of cylinder with given radius
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    bottle_idx = np.sqrt(X**2 + Y**2) < radius

    # Fill in these indices with the values we want
    dims = (N, N)
    PDs = np.zeros(dims)
    T1s = np.zeros(dims)
    T2s = np.zeros(dims)
    PDs[bottle_idx] = PD
    T1s[bottle_idx] = T1
    T2s[bottle_idx] = T2

    # If the user didn't specify an off-resonance range, then they
    # don't want any off-resonance
    if df_range is None:
        df = np.zeros(dims)
    else:
        # Make a simple off-resonance map, linear gradient
        fx = np.linspace(df_range[0], df_range[1], N)
        fy = np.zeros(N)
        df, _ = np.meshgrid(fx, fy)

    return(PDs, T1s, T2s, df)
