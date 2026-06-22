First = make-model-hal.py
	^ Saves model files as a .yml file in model_files/ directory

(Optional) = ind-llh-profiles.py 
	^ Saves information from individual log-likelihood profiles in ResultsEverything.txt 

Second = adding_sources.py 
	^ Creates individual log-likelihood profiles and adds them to each other 

Functions in stacking-functions.py, need to be loaded into each ^^ files

Full file (not broken into three) old/stacking-seyferts-old.py

Job submission file (to sbatch) is job-stack.sh
