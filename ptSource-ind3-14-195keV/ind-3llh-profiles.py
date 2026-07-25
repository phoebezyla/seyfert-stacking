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
DIR = '/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/seyfert-stacking-git/crabtests'
MAP = os.path.join(DATADIR,'maptree-fhit2pct-pass5f-mlp-chunk1-1510.root')
DR = os.path.join(DATADIR, 'detRes-fhit2pct-pass5f-mlp-refit.root')

## Define energy bins ##
midE = np.array([1.0,5.0,10.0])  # TeV
lowerE = 0.5 * 1e9               # keV
upperE = 100 * 1e9               # keV

## Set index ##
ix = "3"      # index in model files = -ix
index = -2.0

## Load CSV and initialize arrays ##
df = pd.read_csv("data14-195.csv",sep='\\s+').to_numpy()

sourceName = df[:,0]
RA = df[:,1]
Dec = df[:,2]
A = df[:,3]

data_radius = 5.
model_radius = 8.

results_df = {}
IntC = []
valN = 400
log_val = np.zeros(valN)

bins = ['B2C0','B2C1','B3C0','B3C1','B4C0','B4C1','B5C0','B5C1',
                 'B6C0','B6C1','B7C0','B7C1','B8C0',
                 'B8C1','B9C0','B9C1','B10C0','B10C1']

## Begin loop over pivot energies ##
for j, pivot in enumerate(midE):

    ## Begin loop over sources ##
    for i,c in enumerate(sourceName):
        ra = RA[i]
        dec = Dec[i]
        a = A[i]
    
        # Load source model #
        model_file = "%s/models/crab_model_ix%s_%.1fTeV.yml"%(DIR,ix,pivot)
        model = threeML.load_model(model_file)        
    
        # Calculate joint and log likelihoods #
        llh, jl = StackingAnalysis.calc_likelihoods(model,c,ra,dec,bins,MAP,DR)
        param_df, like_df = jl.fit(quiet=True)
    
        print(param_df)
        print(like_df)
   
        # Save results from fit 
        saveResults(llh,jl,c,pivot,ix) 
     
        # Get likelihood profile around min norm
        indminNorm = param_df['value'][0] # in kev-1 s-1 cm-2
    
        norms,log_val,ts = StackingAnalysis.likelihood_profile(indminNorm,jl,param_df,like_df,c,valN=valN)
        
        # Build result DF for source
        results_df[c] = {
            "nullLLH": ts.iloc[0]['Null hyp.'],
            "alt_hyp": ts.iloc[0]['Alt. hyp.'],
            "TS":      ts.iloc[0]['TS'],
            "norms":   norms,
            "log_val": log_val,
        }
    
    
        # Plot individual log Profile #
        IntC.append(IntervalContainer(lowerE[j],upperE[j],norms,log_val,101))
        minlogN=np.log10(indminNorm) - 2
        maxlogN=np.log10(indminNorm) + 8
        
        figname = os.path.join(DIR,"crab_ix%s_%.1fTeV_pllh.png"%(ix,pivot))
        
        plot_logProfile_alt([IntC[-1]],param_df,like_df,c,minlogN=minlogN,\
            maxlogN=maxlogN,show=False,save=figname)
    
        ## End Source loop ##
        #####################
    
    # Save results dataframe to csv #
    # creates one results csv for each index & pivot energy #
    f = open("%s/results-%six-%.1fTeV-individual.csv"%(DIR,ix,pivot),"w",newline="")
    writer = csv.writer(f)
    writer.writerow(["sourceName","nullLLH","alt_hyp","TS"] + \
        [f"norms_{i}" for i in range(valN)] + \
        [f"log_val_{i}" for i in range(valN)])
    for source, vals in results_df.items():
        writer.writerow([source, vals["nullLLH"], vals["alt_hyp"], \
            vals["TS"]] + list(vals["norms"]) + list(vals["log_val"]))
    f.close()
