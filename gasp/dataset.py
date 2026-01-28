import os
import logging
import numpy as np
from gasp import dataloader, get_project_path

logger = logging.getLogger(__name__)

# =============================================================================
# Dataset Configuration
# =============================================================================

DATASETS = {
    'phantom_0': {
        'name': 'phantom_0',
        'url': 'https://drive.google.com/file/d/1ttvKQUAPjRJdtocAIJzUv1_7n3kBEOSI/view?usp=share_link',
        'path': '20190401_GASP_PHANTOM',
        'subfolder': None,
        'files': [
            'meas_MID48_TRUFI_NBPM_2019_02_27_FID41503.dat',
            'meas_MID49_TRUFI_NBPM_2019_02_27_FID41504.dat',
            'meas_MID50_TRUFI_NBPM_2019_02_27_FID41505.dat'
        ],
        'description': 'GASP Phantom - April 2019'
    },
    'phantom_1': {
        'name': 'phantom_1',
        'url': 'https://drive.google.com/file/d/1ttvKQUAPjRJdtocAIJzUv1_7n3kBEOSI/view?usp=share_link',
        'path': '20190401_GASP_PHANTOM',
        'subfolder': None,
        'files': [
            'meas_MID54_TRUFI_NBPM_2019_02_27_FID41509.dat',
            'meas_MID55_TRUFI_NBPM_2019_02_27_FID41510.dat',
            'meas_MID56_TRUFI_NBPM_2019_02_27_FID41511.dat'
        ],
        'description': 'GASP Phantom - April 2019 (alt)'
    },
    'water_long_tr': {
        'name': 'water_long_tr',
        'url': 'https://drive.google.com/file/d/1l-JqXUnn7WVubMRaSI1uVsenuQg1MUeS/view?usp=share_link',
        'path': '20190507_GASP_LONG_TR_WATER',
        'subfolder': None,
        'files': [
            'meas_MID12_TRUFI_TE12_FID42712.dat',
            'meas_MID13_TRUFI_TE24_FID42713.dat',
            'meas_MID14_TRUFI_TE48_FID42714.dat'
        ],
        'description': 'Long TR Water - May 2019'
    },
    'crisco_water': {
        'name': 'crisco_water',
        'url': 'https://drive.google.com/file/d/1k4NIhvuCBZKDNKcF4aFhtXEL2JbKw8VU/view?usp=sharing',
        'path': '20190506_GASP_CRISCO_WATER_PHANTOMS',
        'subfolder': None,
        'files': [
            'meas_MID22_TRUFI_TE12_FID42700.dat',
            'meas_MID24_TRUFI_TE24_FID42702.dat',
            'meas_MID25_TRUFI_TE48_FID42703.dat'
        ],
        'description': 'Crisco Water Phantoms - May 2019'
    },
    'water_fat': {
        'name': 'water_fat',
        'url': 'https://drive.google.com/file/d/1yxmriHAoNubNtyZDPak-TmM29tBHDmFv/view?usp=sharing',
        'path': '20190508_GASP_WATER_FAT_PHANTOM',
        'subfolder': None,
        'files': [
            'meas_MID28_TRUFI_TE3_FID42728.dat',
            'meas_MID31_TRUFI_TE6_FID42731.dat',
            'meas_MID33_TRUFI_TE12_FID42733.dat'
        ],
        'description': 'Water Fat Phantom - May 2019'
    },
    'knee_invivo': {
        'name': 'knee_invivo',
        'url': 'https://drive.google.com/file/d/1O8xm9yWk-3vA8H90d3bVer7Ec8u143G-/view?usp=sharing',
        'path': '20190812_GASP_INVIVO_Sag_Knee',
        'subfolder': None,
        'files': [
            'meas_MID131_TRUFI_TE3_FID48578.dat',
            'meas_MID132_TRUFI_TE6_FID48579.dat',
            'meas_MID133_TRUFI_TE12_FID48580.dat'
        ],
        'description': 'In-vivo Sagittal Knee - Aug 2019'
    },
    'brain_hip': {
        'name': 'brain_hip',
        'url': 'https://drive.google.com/file/d/18NV--KkY9QmXm9OVSL73Lvm8Iqsw7ALH/view?usp=sharing',
        'path': '20190827_GASP_INVIVO_BRAIN_HIP',
        'subfolder': None,
        'files': [
            'meas_MID299_TRUFI_TE3_FID49324.dat',
            'meas_MID300_TRUFI_TE6_FID49325.dat',
            'meas_MID301_TRUFI_TE12_FID49326.dat'
        ],
        'description': 'In-vivo Brain Hip - Aug 2019'
    },
    'phantom_2023_fa90': {
        'name': 'phantom_2023_fa90',
        'url': 'https://drive.google.com/file/d/11szQZR8MPmT09zaM-lCSc4nUlNC4E-el/view?usp=sharing',
        'path': '20231106_GASP_PHANTOM',
        'subfolder': None,
        'files': [
            'meas_MID162_bSSFP_gasp_knee_fa90_1x1x2_2D_TR6ms_FID55595.dat',
            'meas_MID163_bSSFP_gasp_knee_fa90_1x1x2_2D_TR12ms_FID55596.dat',
            'meas_MID164_bSSFP_gasp_knee_fa90_1x1x2_2D_TR24ms_FID55597.dat'
        ],
        'description': 'GASP Phantom FA90 - Nov 2023'
    },
    'phantom_2023_fa20': {
        'name': 'phantom_2023_fa20',
        'url': 'https://drive.google.com/file/d/11szQZR8MPmT09zaM-lCSc4nUlNC4E-el/view?usp=sharing',
        'path': '20231106_GASP_PHANTOM',
        'subfolder': None,
        'files': [
            'meas_MID165_bSSFP_gasp_knee_fa20_1x1x2_2D_TR6ms_FID55598.dat',
            'meas_MID166_bSSFP_gasp_knee_fa20_1x1x2_2D_TR12ms_FID55599.dat',
            'meas_MID167_bSSFP_gasp_knee_fa20_1x1x2_2D_TR24ms_FID55600.dat'
        ],
        'description': 'GASP Phantom FA20 - Nov 2023'
    },
    'knee_2023_famax': {
        'name': 'knee_2023_famax',
        'url': 'https://drive.google.com/file/d/1pbn2gsxe-PjvW9vSIE6pnfbCUVWrUGbe/view?usp=sharing',
        'path': '20231106_GASP_KNEE',
        'subfolder': None,
        'files': [
            'meas_MID123_bSSFP_gasp_knee_faMax_1x1x2_2D_TR6ms_FID55556.dat',
            'meas_MID124_bSSFP_gasp_knee_faMax_1x1x2_2D_TR12ms_FID55557.dat',
            'meas_MID125_bSSFP_gasp_knee_faMax_1x1x2_2D_TR24ms_FID55558.dat'
        ],
        'description': 'GASP Knee FA Max - Nov 2023'
    },
    'knee_2023_fa20': {
        'name': 'knee_2023_fa20',
        'url': 'https://drive.google.com/file/d/1pbn2gsxe-PjvW9vSIE6pnfbCUVWrUGbe/view?usp=sharing',
        'path': '20231106_GASP_KNEE',
        'subfolder': None,
        'files': [
            'meas_MID127_bSSFP_gasp_knee_fa20_1x1x2_2D_TR6ms_FID55560.dat',
            'meas_MID128_bSSFP_gasp_knee_fa20_1x1x2_2D_TR12ms_FID55561.dat',
            'meas_MID129_bSSFP_gasp_knee_fa20_1x1x2_2D_TR24ms_FID55562.dat'
        ],
        'description': 'GASP Knee FA20 - Nov 2023'
    },
    'knee_2023_dixon': {
        'name': 'knee_2023_dixon',
        'url': 'https://drive.google.com/file/d/1pbn2gsxe-PjvW9vSIE6pnfbCUVWrUGbe/view?usp=sharing',
        'path': '20231106_GASP_KNEE',
        'subfolder': None,
        'files': ['meas_MID126_DIXON_2D_3echoes_FID55559.dat'],
        'description': 'GASP Knee Dixon - Nov 2023'
    },
    'phantom_dec2023': {
        'name': 'phantom_dec2023',
        'url': 'https://drive.google.com/file/d/10ZRAlIO9w5Q3EsJLHXpnIwU14HM9n12L/view?usp=sharing',
        'path': '20231222_GASP_PHANTOM',
        'subfolder': None,  # Set dynamically: 'dixon', 'gasp_fa20', 'gasp_fa90'
        'files': None,  # Auto-detect from directory
        'description': 'GASP Phantom - Dec 2023'
    },
    'ankle_dec2023': {
        'name': 'ankle_dec2023',
        'url': 'https://drive.google.com/file/d/10hHegaWbiDYv4MsOt8nXDLpxl1b1xccE/view?usp=sharing',
        'path': '20231222_GASP_ANKLE',
        'subfolder': None,  # Set dynamically: 'dixon', 'fa20', 'fa90'
        'files': None,  # Auto-detect from directory
        'description': 'GASP Ankle - Dec 2023'
    },
    'phantom_mar2024': {
        'name': 'phantom_mar2024',
        'url': 'https://drive.google.com/file/d/1M1WdParsJlWMd5es3ve_e3_lfjS9O-Bl/view?usp=sharing',
        'path': '20240312_GASP_PHANTOM',
        'subfolder': None,  # Set dynamically: 'dixon', 'fa20', 'fa90'
        'files': None,  # Auto-detect from directory
        'description': 'GASP Phantom - March 2024'
    },
    'phantom_mar2024_27': {
        'name': 'phantom_mar2024_27',
        'url': 'https://drive.google.com/file/d/1kxqMLtBhsXH0DN4UbgCYwIkjRbSM2CMA/view?usp=sharing',
        'path': '20240327_GASP_PHANTOM',
        'subfolder': None,  # Set dynamically: 'dixon', 'fa20', 'fa90', 'dixon2'
        'files': None,  # Auto-detect from directory
        'description': 'GASP Phantom - March 27, 2024'
    },
}

# =============================================================================
# Core Functions
# =============================================================================

def load_dataset(name=None, url=None, path=None, subfolder=None, files=None,
                 description=None, base_path=None, filter=None):
    """
    Load a dataset by name or by providing dataset parameters directly.

    Can be called as:
        load_dataset('phantom_0')  # by name
        load_dataset(**DATASETS['phantom_0'])  # unpacking a dataset dict
        load_dataset(url='...', path='folder_name')  # direct parameters
        load_dataset('phantom_dec2023', subfolder='gasp_fa20', filter='fa20')

    Args:
        name: Dataset name (key in DATASETS) or Google Drive URL (for backwards compat)
        url: Google Drive URL for the dataset
        path: Folder name within the data directory (e.g., '20190401_GASP_PHANTOM')
        subfolder: Optional subfolder within the dataset folder
        files: Optional list of specific files to load
        description: Dataset description (ignored, for dict unpacking compatibility)
        base_path: Base path for data storage. Defaults to project path.
        filter: Optional string to filter files. Only files containing this string
                will be loaded. Applied to both explicit file lists and auto-detected files.

    Returns:
        np.ndarray: Stacked data array with shape (..., n_files)
    """
    if base_path is None:
        base_path = get_project_path()

    # Check if name is a known dataset key
    if name in DATASETS:
        config = DATASETS[name]
        url = url or config['url']
        path = path or config['path']
        files = files if files is not None else config.get('files')
        subfolder = subfolder if subfolder is not None else config.get('subfolder')
    elif name is not None and url is None:
        # Backwards compat: name could be a URL
        url = name

    # Download data
    if (url is not None):
        dataloader.download_data(url, path, base_path)

    # Build filepath
    if subfolder:
        filepath = os.path.join(base_path, 'data', path, subfolder)
    else:
        filepath = os.path.join(base_path, 'data', path)

    # Get files (from config or directory listing)
    if files is None:
        files = sorted(os.listdir(filepath))

    # Apply filter if specified
    if filter is not None:
        logger.debug(f'Filter: {filter}')
        files = [f for f in files if filter in f]

    logger.debug(f'Loading from: {filepath}')
    logger.debug(f'Files: {files}')

    # Load and stack
    data_list = [
        dataloader.read_rawdata(os.path.join(filepath, f))['data']
        for f in files
    ]

    return np.stack(data_list, axis=-1)


def list_datasets():
    """Return available datasets with descriptions."""
    return {k: v.get('description', '') for k, v in DATASETS.items()}


# =============================================================================
# Backward-Compatible Aliases
# =============================================================================

def load_dataset0(base_path=None):
    return load_dataset('phantom_0', base_path=base_path)


def load_dataset1(base_path=None):
    return load_dataset('phantom_1', base_path=base_path)


def load_dataset2(base_path=None):
    return load_dataset('water_long_tr', base_path=base_path)


def load_dataset3(base_path=None):
    return load_dataset('crisco_water', base_path=base_path)


def load_dataset4(base_path=None):
    return load_dataset('water_fat', base_path=base_path)


def load_dataset5(base_path=None):
    return load_dataset('knee_invivo', base_path=base_path)


def load_dataset6a(base_path=None):
    return load_dataset('brain_hip', base_path=base_path)


def load_dataset7a(base_path=None):
    return load_dataset('phantom_2023_fa90', base_path=base_path)


def load_dataset7b(base_path=None):
    return load_dataset('phantom_2023_fa20', base_path=base_path)


def load_dataset8a(base_path=None):
    return load_dataset('knee_2023_famax', base_path=base_path)


def load_dataset8b(base_path=None):
    return load_dataset('knee_2023_fa20', base_path=base_path)


def load_dataset8c(base_path=None):
    return load_dataset('knee_2023_dixon', base_path=base_path)


def load_dataset9a(base_path=None):
    """Retrieves GASP Phantom data for dixon - Experiment from Dec 22, 2023"""
    return load_dataset('phantom_dec2023', base_path=base_path, subfolder='dixon')


def load_dataset9b(base_path=None):
    """Retrieves GASP Phantom data for fa20 - Experiment from Dec 22, 2023"""
    return load_dataset('phantom_dec2023', base_path=base_path, subfolder='gasp_fa20')


def load_dataset9c(base_path=None):
    """Retrieves GASP Phantom data for fa90 - Experiment from Dec 22, 2023"""
    return load_dataset('phantom_dec2023', base_path=base_path, subfolder='gasp_fa90')


def load_dataset10a(base_path=None):
    """Retrieves GASP Ankle data for dixon - Experiment from Dec 22, 2023"""
    return load_dataset('ankle_dec2023', base_path=base_path, subfolder='dixon')


def load_dataset10b(base_path=None):
    """Retrieves GASP Ankle data for fa20 - Experiment from Dec 22, 2023"""
    return load_dataset('ankle_dec2023', base_path=base_path, subfolder='fa20')


def load_dataset10c(base_path=None):
    """Retrieves GASP Ankle data for fa90 - Experiment from Dec 22, 2023"""
    return load_dataset('ankle_dec2023', base_path=base_path, subfolder='fa90')


def load_dataset11(base_path=None, foldername='dixon'):
    """Retrieves GASP Phantom data - Experiment from March 12, 2024"""
    return load_dataset('phantom_mar2024', base_path=base_path, subfolder=foldername)


def load_dataset11a(base_path=None):
    return load_dataset11(base_path, 'dixon')


def load_dataset11b(base_path=None):
    return load_dataset11(base_path, 'fa20')


def load_dataset11c(base_path=None):
    return load_dataset11(base_path, 'fa90')


def load_dataset12(base_path=None, foldername='dixon'):
    """Retrieves GASP Phantom for dixon - Experiment from March 27, 2024"""
    return load_dataset('phantom_mar2024_27', base_path=base_path, subfolder=foldername)


def load_dataset12a(base_path=None):
    return load_dataset12(base_path, 'dixon')


def load_dataset12b(base_path=None):
    return load_dataset12(base_path, 'fa20')


def load_dataset12c(base_path=None):
    return load_dataset12(base_path, 'fa90')


def load_dataset12d(base_path=None):
    return load_dataset12(base_path, 'dixon2')
