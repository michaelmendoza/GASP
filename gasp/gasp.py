'''GASP module.'''

import numpy as np

def gasp(I, D, pc_dim=0):
    '''Generation of Arbitrary Spectral Profiles.

    Parameters
    ----------
    I : array_like
        Array of phase-cycled images.
    D : array_like
        Vector of samples of desired spectral profile.
    pc_dim : int, optional
        Axis containing phase-cycles.

    Returns
    -------
    I0 : array_like
        Combined image with spatial response approximating D.
    '''

    # Let's put the phase-cycle dimension last
    I = np.moveaxis(I, pc_dim, -1)

    # Save the in-plane dimsensions for reshape at end
    xx, yy = I.shape[:2]

    # Now let's put all the voxels' time curves down the first dim
    I = I.reshape((-1, I.shape[-1]))

    # Now repeat the desired spectral profile the correct number of
    # times to line up with the length of each column
    D = np.tile(D, (int(I.shape[0]/D.size),))
    # print(I.shape, D.shape)

    # Now solve the system
    x = np.linalg.lstsq(I, D, rcond=None)[0]
    # print(x.shape)

    return I.dot(x).reshape(xx, yy)


if __name__ == '__main__':
    pass
