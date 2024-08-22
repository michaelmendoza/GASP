
### Discusion Notes

## Notes / Questions?

- Discovered we aren't using a full period for SSFP i.e. 4 pi
- Investigate complex valued forcing function
- If the field is strong, the banding frequency is increased.  If the resolution is too low, multiple bands will be contained in a single voxel leading to bad artifacts and poor performance of GASP

## Things to do:

-[x] Create periodic forcing function
-[x] Mask things outside of a period so it isn't included in the calculation
-[x] Could also only consider the center line
-[x] Create simulation data and mimics phantom
-[x] Use field map from phantom data in simulation data
-[ ] Try phase unwrapping
-[ ] Remove phase that accruals over time  

## Ideas for Future Work
- Using a combinations of GASP models in some kind of ensemble filter, perhaps with classifying tissues to be associated with a GASP filter or classfying to set the weights of the filter
- Normalize by GS recon?

## New ideas
- Match filter function for tissue selection
- Finding min spectal size
- MR spectroscopy

## Things to investigate
- Window 
    - Gaussian
    - Others
- Number of TRs and Phase-Cycles vs fitting (MSE)
- Optimize TR/PC for Orthogonality (metrics for Orthgonality)
- TR (Modes) vs TR (other values)
- Train model on phantom (one period) vs train on simulated phantom (one train)
- Train model with simulated - and use on real data for water/fat separation
- FEMR for generating data 
- using SSFP instead of bSSFP
- sensentivity to steady state 
- train with a range of flip-angles
- gasp needs zero pad of at least 1-2 values to work, why?
- Multiple alpha, vs different tissues 
- Single line GASP
- Mult alpha data for GASP
- Deep Network
- Test real 
- Nonlinear regression
- Reduced field of view with gasp 
- Plot of coefficents, and look at residual 
- Can gasp be used for segmentation?
- Create plot angle vs angle (off)
- Validation of Spatial vs Spectral Profile
- Simulation training with Evaluating Real Data
- DL / Non-linear LeastSqaures
- Phase for forcing functions 
- Look at Ellipses 
- Multiple Fat peaks
- Spectroscopy 
- GASP per coil or afer coil combine 
- Train on one gaussian at two locations, and add co-efficeints and see if i get the two dual filter 
- Can it have an impulse response (Get spectrum point spread function and get response of delta function at every point)
- Question: Does a fft of the off resonance spectrum mean anything? Can it be useful?
- Use ray to paralize compute things 
- Is there such a thing as a spectral filter, like to remove ocilation in the spectrum?

Discussion: 
- single zero crossing allowed for linear model 
- The most orthogonal set, what is a measure of how good or effiecent of basis set, measure dimensionstiy, measure of linear-independance, RANK, SVD of, span of desired forcing fuctions and see how well your basis spans that space 

----
Paper:

Analysis:
    N [] Linear LSE vs Non-linear LSE

    [] Optimization: Minimum number of to make GASP work. How do a pick a reasonable set?

        M [\] MSE & Plots at with input images (i.e. different TRs and PCs) - 3 TRs & 16 PCs, 3 TR & 8 PCs, Brute Fource

        N [\] Automatic Parameter Optimization/Selection with some algo

    N []  Phase Effects: Elliptical Relationships. Does GASP preserve ellipses, continuity of phase?

    N [] Difference of off-resonance profiles from phase cycling and linear gradient. Assumption: Off-resonace in linear gradient is good estimate of phase cycling off-resonance profiles

    M [\] Training on simlulated vs real phantoms: Effectiveness

    M [] Investigation of different windowing functions
        - Window function with phase? Linear phase? Maybe zero phase is wrong? 

    M [\] Training with muliple tip angles. 

    M [X] Sensitivity analysis - Training at one tip angle, and evaluting at a different tip angle 

    M [ ] Linear combinations of GASP co-effiecenit. IS GASP a linear operator? Does GASP perserve phase? 
        - Check GASP per coil or afer coil combine 


---
8/22/2024:
- Plot of alpha vs T1/T2 and spectra for different alpha and T1/T2  
- Look at condition number of matrix
- See if water vs fat, switch and see if its symmetric 
- What is low alpha better than high alpha??
- Look at FFT of SSFP signal
- Look into l curve method
- Look to regularization (L1, L2, etc)
- Idea: High tip angle leads to more circular vs low tip angle. And therefore low tip angle has more information.