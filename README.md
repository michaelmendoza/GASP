
# GASP: Generation of Arbitrary Spectral Profiles

## Overview

GASP (Generation of Arbitrary Spectral Profiles) is a Python library for simulating and analyzing MRI sequences, particularly focused on balanced Steady-State Free Precession (bSSFP) and spectral shaping techniques. This project provides tools for simulating MRI signals, generating phantoms, and applying the GASP method to achieve desired spectral profiles.

## Features

- Simulation of bSSFP sequences
- Generation of various phantom types (e.g., Shepp-Logan, circles, blocks)
- Implementation of the GASP method for spectral shaping
- Tools for analyzing and visualizing MRI data
- Support for different tissue types and their relaxation properties

## Development

This project requires python 3.8+ and has the following dependancies: 
`numpy matplotlib scikit-image seaborn pymapvbvd jupyterlab gdown scipy`

To setup a local python enviroment with conda:

Create a new conda environment from scatch 
> ```
> conda create -n gasp python=3.8 
> conda activate gasp
> ```
> Then install packages with pip:
> ```
> pip install numpy matplotlib scikit-image seaborn pymapvbvd jupyterlab gdown scipy
> ```

## Usage

Here's a basic example of how to use the GASP simulation:

```python
from gasp import simulation, responses

# Set up simulation parameters
width, height = 256, 256
npcs = 16
TRs = [5e-3, 10e-3, 20e-3]
alpha = np.deg2rad(60)
gradient = 2 * np.pi

# Create a desired spectral profile
D = responses.gaussian(width, bw=0.2, shift=0)

# Simulate GASP
Ic, M, An = simulation.simulate_gasp(D, width, height, npcs, TRs, alpha, gradient)

# Visualize results
simulation.view_gasp_results(Ic, M, D)
```

## Modules

- `analysis.py`: Contains functions for analyzing GASP results and Dixon methods
- `dataloader.py`: Handles loading of raw MRI data
- `dataset.py`: Provides functions to load specific datasets
- `gasp.py`: Core implementation of the GASP method
- `phantom.py`: Functions for generating various phantom types
- `responses.py`: Implements different spectral response functions
- `simulation.py`: Main simulation routines for bSSFP and GASP
- `ssfp.py`: Implementation of Steady-State Free Precession signal equations
- `tissue.py`: Defines tissue properties and generates tissue phantoms
- `view.py`: Visualization tools for 3D data
