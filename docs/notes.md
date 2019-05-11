
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
