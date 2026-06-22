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
DATADIR = '/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/'
MODELDIR = os.path.join(DATADIR, 'ptSource-ind3-14-195keV/model_files/')

lowerE = np.logspace(np.log10(0.5),np.log10(10),6)[:5] 
midE = np.power(10,np.log10(lowerE)+0.25)
e = midE[0]

# Create source model #
fjl, clm = StackingAnalysis.ptsource_model("finalNorm",0,0,1,e,ind=3,Kmax=None)
name="clm"

clm.save('%s/%s_modelFile.yml'%(MODELDIR,name),overwrite=True) 
    


