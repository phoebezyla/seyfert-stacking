A repository for P. Zyla's seyfert-stacking analysis. Each sub-directory is named according to the conditions it and its contents were made under, and contain resultant plots, outputs, and the `stacking-seyferts.py` scripts that was ran. 


Recent changes: 
Made an excel sheet for testing crab values. I’ve noticed that for both index -2.0 and -2.7, the TS values from the stacking match the TS values from the individual profile for ONLY pivot = 1 TeV. With a higher pivot energy, the stacked TS separates further from the individual TS (when stacking with only one crab.) The TS, log likelihood, and normalization values should (I believe) be the same when comparing the individual profile of the crab to the stacked (where the stacked list is only the crab). 
Our energy band is so large now (and we are using all energies available to HAWC), so CastroLike’s dependency on the pivot energy is now much more obvious. For Hugo’s analysis, where the analysis energy bands (start and stop for IntervalContainer) were narrow and varied with the pivot energy, you wouldn’t see this effect – it is only because we have expanded the energy range that we are seeing it. 
Other evidence: look at how expected_flux and norms are treated within CastroLike.py. `norms` is an array of 200 parameter values – the Interval Container object takes those parameter values (and the 200 likelihood values) and extrapolates a smooth curve such that we can grab a likelihood value for any normalization, not just the 200 input ones. 
An array of x values between START and STOP is generated. An array of corresponding y values is generated with get_total_flux (which comes from the likelihood model, which we set to be clm.) `get_total_flux` sums the flux contributions from all point sources within a model at each energy (xx) and returns an array the same length as xx. It returns the differential photon flux in kev-1 s-1 cm-2 at each energy value. 
Between START and STOP, the xx and yy arrays are integrated via simps. This is valid for Hugo’s case, where the input parameter values are the average flux over an energy band. Those average flux values are integrated over START – STOP and divided by the (narrow) band width, to get the model’s predicted average flux (same units as input param array). 
For my case, we are inputting the flux normalization at the pivot energy (and NOT a band-averaged flux) – for my norms to be equal to what Hugo’s code wants, we need to evaluate at the pivot directly.  
Changes: 
-	Weighted nullLLH when appending to array
-	Added weight and pivot properties to interval container object
-	Modified get_log_like function to be compatible with normalization parameter 
o	No longer integrating – calculate expected_flux at the pivot energy (instead of over a narrow band of energies)
o	Weight the likelihood instead of the expected flux (brighter sources contribute more to the likelihood profile)
Made these changes after comparing the following to the individual crab likelihood profile:
-	Stacking one crab
-	Stacking two crabs, each with a weight of 0.5
-	Stacking two crabs, one 0.25 and one 0.75

I’ve now modified the cl.plot() function, to try and make it a little more useful for me. It has not significantly improved. 

############################################

Looking at the TS values, I feel suspicious
	Index = 2.0	Index = 2.7	Index = 3.0
Pivot = 1 TeV	TS = -0.038014	TS = -0.001218	24.021552
Pivot = 5 TeV	TS = -0.014421	TS = 7.745532	284.529755
Pivot = 10 TeV	TS = 3.334126	TS = 128.230567	36.683536

There are some sources that, when fitting, have best-fit K values at or below the minimum K value I have set currently (especially as pivot gets higher and index gets steeper)
Options: exclude from stacking, treat as upper limits (only upper bound on normalizxation contributes to the stacked likelihood) (this is done already in get_measurement, but the spline in IntervalContainer might be messing up still), or cut off the norms range to ensure that the IntC spline is only built over physically meaningful K values (and not ~= 0 values)
At too small a value for K, the likelihood profile is flat (no sensitivity in data), so the minimizer will be pinned no matter what (can’t just move the boundary)

Change: 
-	Modified likelihood_profile function (in stacking_functions.py) to have a set, fixed range to scan over the norms 
o	This is instead of `norms = np.linspace(np.log10(indminNorm)-5,np.log10(indminNorm)+5,valN)`, which changes the range for a given source depending on its normalization

