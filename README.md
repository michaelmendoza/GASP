# GASP

GASP is an MRI technique for the Generation of Arbitary Spectral Profiles using Orthonormal Basis Combination of bSSFp MRI.

## Development

This project requires python 3.8+ and has the following dependancies: 
numpy, matplotlib, scikit-image, mapvbvd, and jupyterlab.

To setup a python enviroment with conda:

1. Create a new conda environment from scatch 
> ```
> conda create -n gasp python=3.8 
> conda activate gasp
> ```
> Then install packages with pip:
> ```
> pip install numpy matplotlib scipy seabornpy mapvbvd jupyterlab gdown
> ```