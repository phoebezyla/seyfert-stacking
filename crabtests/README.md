# Crab Notes 

## Summary of Changes

## Issues and Fixes

1. Running the stacking script with one crab did not /always/ return the same TS as the individual analysis. The stacked TS matched the individual TS when the pivot energy = 1 TeV. With a higher pivot energy, the stacked TS separated further form the individual TS. 

CastroLike allows the user to set an energy band (over which the analysis is performed) when determining the start and stop IntC values. These are the energy values over which the flux is integrated to get expected_flux (and then an individual source's log likelihood.) This integration works well when working with brighter sources. 

For my dimmer, higher energy sources, we can instead evaluate the log likelihood directly at the pivot energy. 

This is a modification of the `get_log_like` function within the `CastroLike.py` script.

2. When running the stacking script with two crabs (both with a relative weight of 0.5), the returned TS was not equal to an individual crab's TS. The weights were not being applied properly -- the `interval_container.weight` attribute was multiplying the `expected_flux` value instead of the likelihood itself. 

The individual source likelihood is now multiplied by the relative weight before being added to the population's stacked likelihood. This is a modification of the `get_log_like` function within the `CastroLike.py` script. 

3. The TS values for two crabs (both with a relative weight of 1.0) did not equal 2 * individual crab TS. 

The TS for the stacking analysis is calculated differently than the TS for the individual analysis. For the stacked profile, $TS_stacked = 2 * (totalnull - stacked_log_like)$. The weight needs to be applied to both the stacked likelihood and the total null variables. 

When a source is loaded, its null estimation is now multiplied by the source's weight before contributing to the `totalnull` variable. This is a modification of the `adding-sources.py` script.  
