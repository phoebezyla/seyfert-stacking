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
MODELDIR = os.path.join(DIR,'model_files/')
MAPDIR = os.path.join(MODELDIR,'hd5_files/')

MAP = os.path.join(DATADIR,'maptree-fhit2pct-pass5f-mlp-chunk1-1510.root')
DR = os.path.join(DATADIR, 'detRes-fhit2pct-pass5f-mlp-refit.root')

## Define energy bins ##
lowerE = np.logspace(np.log10(0.5),np.log10(10),6)[:5]  # three vals 500 GeV to 10 TeV
midE = np.power(10,np.log10(lowerE)+0.25)
upperE = np.power(10,np.log10(lowerE+0.5))

## Only do one pivot energy value
midE = [np.array(midE[0])]

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

bestGuess = 1e-26
IntC = []
nullLLH = []
TS = []
log_val = np.zeros(200)

bins = ['B2C0','B2C1','B3C0','B3C1','B4C0','B4C1','B5C0','B5C1',
                 'B6C0','B6C1','B7C0','B7C1','B8C0',
                 'B8C1','B9C0','B9C1','B10C0','B10C1']

## Begin loop over energy bins ##
for j, e in enumerate(midE):
    
    ## Begin loop over sources ##
    for i,c in enumerate(sourceName):
        ra = RA[i]
        dec = Dec[i]
        a = A[i]

        roi = HealpixConeROI(data_radius=data_radius,model_radius=model_radius,ra=ra,dec=dec)

        # Read source model #
        model_file = "%s/%s_modelFile.yml"%(MODELDIR,c)
        model = threeML.load_model(model_file)        
        print("Loaded YML File: %s"%(c))


        # Make hd5 of HAL model #
        like = HAL("HAWC", MAP, DR, roi)
        like.set_active_measurements(bin_list=bins)
        like.set_model(model)
        like.get_log_like()        

        mapname = "%s/%s_model_map.hd5"%(MAPDIR,c)
        like.write_model_map(mapname, poisson_fluctuate = False)
        print("Wrote hd5 file: %s"%(c))        
