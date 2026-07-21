from astropy import units as u
import matplotlib.pyplot as plt
import os, sys, time
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
import scipy.optimize
import pickle
from astromodels import clone_model
import math
import warnings
import yaml
import numpy as np
import pandas as pd
from stacking_functions import *


## Set paths ##
DATADIR = '/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/seyfert-stacking-git/crabtests'

## Define energy bins ##
midE = np.array([1.0,5.0,10.0])  #TeV

## Index Arrays ##
inds = np.array([-2.0,-2.7,-3.0])
indnames = np.array(['2','27','3'])

for j,ix in enumerate(inds): 
    for e in midE:
        # Create source model #
        fjl, clm = StackingAnalysis.ptsource_model("finalNorm",0,0,1,e,ind=ix)
        clm.save('%s/clm_E_%.1f_TeV_ind%s_modelFile.yml'%(DATADIR,e,indnames[j]),overwrite=True) 
        
    
    
