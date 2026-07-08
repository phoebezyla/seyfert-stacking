from astropy import units as u
import matplotlib.pyplot as plt
import os, sys, time
import csv
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
midE = np.array([1.0,5.0,10.0]) #TeV

## Load CSV and initialize arrays ##
df = pd.read_csv("data14-195.csv",sep='\\s+').to_numpy()

numsources = 51
sourceName = df[:numsources,0]
RA = df[:,1]
Dec = df[:,2]
A = df[:,3]

Atotal = 0
for c in range(len(sourceName)):
    Atotal += A[c]

data_radius = 5.
model_radius = 8.

results_df = {}

IntC_arr = []
TS_arr = []
nullLLH_arr = []


bins = ['B2C0','B2C1','B3C0','B3C1','B4C0','B4C1','B5C0','B5C1',
                 'B6C0','B6C1','B7C0','B7C1','B8C0',
                 'B8C1','B9C0','B9C1','B10C0','B10C1']

## Begin loop over pivot energies ##
for j, e in enumerate(midE):

    ###############################
    ## Perform Bayesian Analysis ##
    
    # Load source model #
    clm_model_file = "%s/model_files/yml_clm/clm_ind%s_modelFile.yml"%(DIR)
    clm = threeML.load_model(clm_model_file)
    datalist = Datalist(clm)

    credInt,uplim = bayesian_ana(model_source,clm,datalist,e,DIR,'finalNorm')


## Save full results dictionary to one csv per index ##
f = open("%s/results-bay-%six.csv"%(DIR,indnames[k]),'w',newline='')
writer = csv.writer(f)
writer.writerow(['pivot','credInt','uplim'])

for pivot,vals in results.items():
    writer.writerow([pivot, vals['credInt'], vals['uplim']])
f.close()
