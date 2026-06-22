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
MODELDIR = os.path.join(DIR,'model_files/yml_initial/')
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

sourceName = df[:1,0]
RA = df[:,1]
Dec = df[:,2]
A = df[:,3]

Atotal = 0
for c in range(len(sourceName)):
    Atotal += A[c]

data_radius = 5.
model_radius = 8.

results_df = {}
IntC = []
log_val = np.zeros(200)

bins = ['B2C0','B2C1','B3C0','B3C1','B4C0','B4C1','B5C0','B5C1',
                 'B6C0','B6C1','B7C0','B7C1','B8C0',
                 'B8C1','B9C0','B9C1','B10C0','B10C1']


## Begin loop over sources ##
for i,c in enumerate(sourceName):
    ra = RA[i]
    dec = Dec[i]
    a = A[i]

    # Load source model #
    model_file = "%s/%s_modelFile.yml"%(MODELDIR,c)
    model = threeML.load_model(model_file)        
    print("Loaded YML File: %s"%(c))

    # Calculate joint and log likelihoods #
    print("Calculating likelihoods...")
    llh, jl = StackingAnalysis.calc_likelihoods(model,c,ra,dec,bins,MAP,DR)
    
    #jl.set_minimizer("ROOT")
    param_df, like_df = jl.fit(quiet=True)
   
    # Save results from fit 
    #     jl results FITS file, jl results yml file 
    #     counts/residuals png, spectrum png, planes png, stacked png
    saveResults(llh,jl,c) 
 
    # Get likelihood profile around min norm
    indminNorm = param_df['value'][0] # in kev-1 s-1 cm-2

    norms,log_val,ts = StackingAnalysis.likelihood_profile(indminNorm,jl,param_df,like_df,c)
    
    # Build result DF for source
    results_df[c] = {
        "nullLLH": ts.iloc[0]['Null hyp.'],
        "alt_hyp": ts.iloc[0]['Alt. hyp.'],
        "TS":      ts.iloc[0]['TS'],
        "norms":   norms,
        "log_val": log_val,
    }


    # Plot individual log Profile #
    IntC.append(IntervalContainer(i+1,i+2,norms,log_val,101))
    minlogN=np.log10(indminNorm) - 2
    maxlogN=np.log10(indminNorm) + 8
    
    figname = os.path.join(DIR,"plots/%s_pllh_2.png"%(c))
    
    #plot_logProfile(IntC[-1],xbest,minllh,save=figname)
    plot_logProfile_alt([IntC[-1]],param_df,like_df,c,minlogN=minlogN,\
        maxlogN=maxlogN,show=False,save=figname)

    ## End Source loop ##
    #####################

# Save results dataframe to csv #
f = open("%s/ind-results1.csv"%(DIR),"w",newline="")
writer = csv.writer(f)
writer.writerow(["sourceName","nullLLH","alt_hyp","TS"] + \
    [f"norms_{i}" for i in range(200)] + \
    [f"log_val_{i}" for i in range(200)])
for source, vals in results_df.items():
    writer.writerow([source, vals["nullLLH"], vals["alt_hyp"], \
        vals["TS"]] + list(vals["norms"]) + list(vals["log_val"]))
f.close()
