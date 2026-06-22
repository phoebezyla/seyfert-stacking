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
DIR = '/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/ptSource-ind3-14-195keV/'
MAP = os.path.join(DATADIR,'maptree-fhit2pct-pass5f-mlp-chunk1-1510.root')
DR = os.path.join(DATADIR, 'detRes-fhit2pct-pass5f-mlp-refit.root')
MODELDIR = os.path.join(DATADIR, 'ptSource-ind3-14-195keV/model_files/')


## Define energy bins ##
lowerE = np.logspace(np.log10(0.5),np.log10(10),6)[:5]  # three vals 500 GeV to 10 TeV
midE = np.power(10,np.log10(lowerE)+0.25)
upperE = np.power(10,np.log10(lowerE+0.5))

## Only do one pivot energy value
e = np.array(midE[0])


## Load CSV and initialize arrays ##
df = pd.read_csv("data14-195.csv",sep='\\s+').to_numpy()

sourceName = df[:,0]
RA = df[:,1]
Dec = df[:,2]
A = df[:,3]

Atotal = 0
for c in range(len(sourceName)):
    Atotal += A[c]

data_radius = 5.
model_radius = 8.


## Begin loop over sources ##
for i,c in enumerate(sourceName):
    ra = RA[i]
    dec = Dec[i]
    a = A[i]

    # Create source model #
    source, model = StackingAnalysis.ptsource_model(c,ra,dec,a,e)
    
    model.save('%s/%s_modelFile.yml'%(MODELDIR,c),overwrite=True) 
    


