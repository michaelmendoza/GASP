import os
import numpy as np
import matplotlib.pyplot as plt 
from gasp import dataset

if __name__ == "__main__":
    M = dataset.load_dataset0()
    print(M.shape)