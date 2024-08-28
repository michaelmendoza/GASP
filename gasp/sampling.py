import math
import numpy as np
import matplotlib.pyplot as plt

def spiral_sampling(n_points, n_turns, tr_range=(5e-3, 20e-3)):
    theta = np.linspace(0, 2*np.pi*n_turns, n_points)
    r = np.linspace(0, 1, n_points)
    min_TR, max_TR = tr_range
    TR = min_TR + r * (max_TR - min_TR)
    PC = theta % (2*np.pi)
    return TR, PC

def fibonacci_sampling(n_points, tr_range=(5e-3, 20e-3)):
    golden_ratio = (1 + 5**0.5) / 2
    i = np.arange(n_points)
    theta = 2 * np.pi * i / golden_ratio
    r = np.sqrt(i / n_points)
    min_TR, max_TR = tr_range
    TR = min_TR + r * (max_TR - min_TR)
    PC = theta % (2*np.pi)
    return TR, PC

def log_polar_sampling(n_points, tr_range=(5e-3, 20e-3)):
    n_radial = int(np.sqrt(n_points))
    n_angular = n_points // n_radial
    
    min_TR, max_TR = tr_range
    log_min, log_max = np.log(min_TR), np.log(max_TR)
    r = np.exp(np.linspace(log_min, log_max, n_radial))
    theta = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
    R, Theta = np.meshgrid(r, theta)
    TR, PC = R.flatten(), Theta.flatten()
    
    if len(TR) > n_points:
        TR, PC = TR[:n_points], PC[:n_points]
    elif len(TR) < n_points:
        extra = n_points - len(TR)
        TR = np.pad(TR, (0, extra), mode='edge')
        PC = np.pad(PC, (0, extra), mode='edge')
    
    return TR, PC

def grid_sampling(n_points, n_turns, tr_range=(5e-3, 20e-3)):
    min_TR, max_TR = tr_range
    n_tr = n_turns
    n_pc = n_points // n_tr
    
    TRs = np.linspace(min_TR, max_TR, n_tr)
    PCs = np.linspace(0, 2*np.pi, n_pc, endpoint=False)
    
    TR, PC = np.meshgrid(TRs, PCs)
    TR, PC = TR.flatten(), PC.flatten()
    
    if len(TR) > n_points:
        TR, PC = TR[:n_points], PC[:n_points]
    elif len(TR) < n_points:
        extra = n_points - len(TR)
        TR = np.pad(TR, (0, extra), mode='edge')
        PC = np.pad(PC, (0, extra), mode='edge')
    
    return TR, PC

def grid_multiples_sampling(n_points, n_turns, tr_range=(5e-3, 20e-3)):
    min_TR, max_TR = tr_range
    n_tr = n_turns
    n_pc = n_points // n_tr
    
    TRs = min_TR * np.linspace(1, n_turns, n_tr)
    PCs = np.linspace(0, 2*np.pi, n_pc, endpoint=False)
    
    TR, PC = np.meshgrid(TRs, PCs)
    TR, PC = TR.flatten(), PC.flatten()
    
    if len(TR) > n_points:
        TR, PC = TR[:n_points], PC[:n_points]
    elif len(TR) < n_points:
        extra = n_points - len(TR)
        TR = np.pad(TR, (0, extra), mode='edge')
        PC = np.pad(PC, (0, extra), mode='edge')
    
    return TR, PC

def grid_pow2_sampling(n_points, n_turns, tr_range=(5e-3, 20e-3)):
    min_TR, max_TR = tr_range
    n_tr = n_turns
    n_pc = n_points // n_tr
    
    TRs = min_TR * 2 ** np.linspace(0, n_turns, n_tr, endpoint=False)
    PCs = np.linspace(0, 2*np.pi, n_pc, endpoint=False)
    
    TR, PC = np.meshgrid(TRs, PCs)
    TR, PC = TR.flatten(), PC.flatten()
    
    if len(TR) > n_points:
        TR, PC = TR[:n_points], PC[:n_points]
    elif len(TR) < n_points:
        extra = n_points - len(TR)
        TR = np.pad(TR, (0, extra), mode='edge')
        PC = np.pad(PC, (0, extra), mode='edge')
    
    return TR, PC

def grid_TR_sampling(n_points, TRs):
    npcs = math.ceil(n_points / len(TRs))
    PCs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    TR, PC = np.meshgrid(TRs, PCs,  indexing='ij')
    TR, PC = TR.flatten(), PC.flatten()
    TR, PC = TR[:n_points], PC[:n_points]
    return TR, PC