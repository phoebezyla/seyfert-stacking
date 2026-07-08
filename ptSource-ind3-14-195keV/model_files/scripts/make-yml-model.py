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
midE = np.array([1.0,5.0,10.0])  #TeV
lowerE = midE/np.power(10,0.25)
upperE = np.power(10,np.log10(lowerE+0.5))

## Index Arrays ##
inds = np.array([-2.7])
indnames = np.array(['27'])

## Load CSV and initialize arrays ##
df = pd.read_csv("data14-195.csv",sep='\\s+').to_numpy()

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
            
            model.save('%s/yml_ind%s_initial/E_%.1f_TeV/%s_modelFile.yml'%(MODELDIR,indnames[j],e,c),overwrite=True)  
    
