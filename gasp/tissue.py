from typing import Tuple
import numpy as np
from gasp import phantom

# T1/T2 values taken from https://mri-q.com/why-is-t1--t2.html
tissue_map = {
    'none': [0, 0],
    'water': [4, 2],
    'white-matter': [0.6, 0.08],
    'gray-matter': [0.9, 0.1],
    'muscle': [0.9, .05],
    'liver': [0.5, 0.04],
    'fat': [0.25, 0.07],
    'tendon': [0.4, 0.005],
    'proteins': [0.250, 0.001]
}

def get_t1(x: int) -> Tuple[float, float]:
    x = int(x)
    keys = list(tissue_map.keys())
    return tissue_map[keys[x]][0]

def get_t2(x: int) -> Tuple[float, float]:
    x = int(x)
    keys = list(tissue_map.keys())
    return tissue_map[keys[x]][1]

def tissue_generator(fov: int=256, type: str='blocks'):
    """ Generates a tissue phantom with a given shape for a number of coils.
    Args:
        fov: size of image.
        type: type of phantom
    Returns:
        tissue dict
    """
    fov = int(fov)
    keys = list(tissue_map.keys())

    if type == 'shepp_logan':
        img = phantom.shepp_logan_phantom([fov, fov])
    elif type == 'circle':
        img = phantom.circle_phantom([fov, fov])
    elif type == 'circles':
        img = phantom.circle_array_phantom([fov, fov])
    elif type == 'block':
        img = phantom.block_phantom_single(fov)
    elif type == 'blocks':
        img = phantom.block_phantom()
    elif type == 'line':
        img = phantom.line_phantom()
    elif type == 'line-whitematter':
        img = phantom.line_phantom() * 2
    elif type == 'line-graymatter':
        img = phantom.line_phantom() * 3
    elif type == 'line-muscle':
        img = phantom.line_phantom() * 4
    elif type == 'line-fat':
        img = phantom.line_phantom() * 6
    else:
        raise ValueError('Incorrect phantom type')

    mask = (img != 0) * 1

    t1 = list(map(lambda x: get_t1(x), img.flatten()))
    t1 = np.array(t1).reshape(img.shape)

    t2 = list(map(lambda x: get_t2(x), img.flatten()))
    t2 = np.array(t2).reshape(img.shape)

    tissue_phantom = {'mask': mask, 't1': t1, 't2': t2 }
    return tissue_phantom
