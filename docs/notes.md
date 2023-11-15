
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

Discussion: 
- single zero crossing allowed for linear model 

