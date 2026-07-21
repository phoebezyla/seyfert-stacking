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


## Define energy bins ##
midE = np.array([1.0,5.0,10.0])  # TeV
lowerE = 0.5 * 1e-9              # keV
upperE = 100 * 1e-9              # keV

## Index Arrays ##
inds = np.array([-2.0,-2.7,-3.0])
indnames = np.array(['2','27','3'])

## Load CSV and initialize arrays ##
df = pd.read_csv("crab.csv",sep='\\s+').to_numpy()

sourceName = df[:,0]
RA = df[:,1]
Dec = df[:,2]
A = df[:,3]

data_radius = 5.
model_radius = 8.

## Begin loop over indices ##
for j,ind in enumerate(inds):

    ## Begin loop over energies ##
    for e in midE: 
       
        ## Begin loop over sources ##
        for i,c in enumerate(sourceName):
            ra = RA[i]
            dec = Dec[i]
            a = A[i]
        
            # Create source model #
            source, model = StackingAnalysis.ptsource_model(c,ra,dec,a,e,ind=ind)
            
            model.save('%s/crab_model_ix%s_%.1fTeV.yml'%(DIR,indnames[j],e),overwrite=True)  
    
