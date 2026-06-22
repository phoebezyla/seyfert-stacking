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
lowerE = np.logspace(np.log10(0.5),np.log10(10),6)[:5]  # three vals 500 GeV to 10 TeV
midE = np.power(10,np.log10(lowerE)+0.25)
upperE = np.power(10,np.log10(lowerE+0.5))

## Only do one pivot energy value
midE = [np.array(midE[0])]

## Load CSV and initialize arrays ##
df = pd.read_csv("data14-195.csv",sep='\\s+').to_numpy()

numsources = 2
sourceName = df[:numsources,0]
RA = df[:,1]
Dec = df[:,2]
A = df[:,3]

#Atotal = 0
#for c in range(len(sourceName)):
#    Atotal += A[c]

data_radius = 5.
model_radius = 8.

results_df = {}

IntC_arr = []
TS_arr = []
nullLLH_arr = []

#results = []
#resultsLow = [] 
#resultsHigh = []
#ULs = [] 
#TSArray = []
#norms = []
#normsErr = []

bins = ['B2C0','B2C1','B3C0','B3C1','B4C0','B4C1','B5C0','B5C1',
                 'B6C0','B6C1','B7C0','B7C1','B8C0',
                 'B8C1','B9C0','B9C1','B10C0','B10C1']


with open("ind-results-2scs.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        source_name = row["sourceName"]
        results_df[source_name] = {
            "nullLLH": float(row["nullLLH"]),
            "alt_hyp": float(row["alt_hyp"]),
            "TS":      float(row["TS"]),
            "log_val": [float(row[f"log_val_{k}"]) for k in range(200)],
            "norms":   [float(row[f"norms_{k}"]) for k in range(200)],
        }

## Begin loop over pivot energies ##
for j, e in enumerate(midE):
    ## Begin loop over sources ##
    for i,c in enumerate(sourceName):
        ra = RA[i]
        dec = Dec[i]
        a = A[i]
        
        # Load individual profiles 
        nullLLH = results_df[c]["nullLLH"]
        print(f"{c} nullLLH: {nullLLH}") 
        TSind   = results_df[c]["TS"]
        print(f"{c} TS: {TSind}")
        log_val = results_df[c]["log_val"]
        print(f"{c} log_val: {log_val}")
        norms   = results_df[c]["norms"]
        print(f"{c} norms: {norms}")        

        # Get likelihood profile around min Norm? #        
        IntC_arr.append(IntervalContainer(i+1,i+2,norms,log_val,101))
        nullLLH_arr.append(nullLLH)
        TS_arr.append(TSind)

        ## End Source loop ##
        #####################

    ###############################
    ## Stack likelihood profiles ##

    totalnull = np.sum(np.asarray(nullLLH_arr))
    print("\nTotal nullLLH: {}".format(totalnull))
    
    # Load source model #
    clm_model_file = "%s/model_files/clm_modelFile.yml"%(DIR)
    clm = threeML.load_model(clm_model_file)
    print("Loaded clm YML File")   


    fjl, datalist = StackingAnalysis.stacked_likelihood(IntC_arr, clm)
    fjl.set_minimizer("ROOT")

    param_df, like_df = fjl.fit(quiet=True)
    print("Parameter results: %s"%(param_df))
    print("Likelihood results: %s"%(like_df))

    # Built total TS array #
    print("total Alt LLH:", like_df.iloc[1]['-log(likelihood)'])
    print("total Null LLH: ", totalnull)
    TS_stacked =  2 * (totalnull - like_df.iloc[1]['-log(likelihood)'])
    print("Total TS: %s"%(TS_stacked))
    

    ####### Check final loglikelihood profile
    indminNorm = param_df['value'][0]
    minlogN=np.log10(indminNorm) - 1.5
    maxlogN=np.log10(indminNorm) + 1.5
   
    norms_Stack,log_val_Stack,a_Stack = StackingAnalysis.likelihood_profile(indminNorm,fjl,param_df,like_df,"Stacked",computeTS=False)
 
    IntC = IntervalContainer(i+1,i+2,norms_Stack,log_val_Stack,101)
    #if indminNorm<normalization.K.min_value:
    #    normalization.k.min_value = 1e-40    

    # Plot stacked profile #    
    figname = os.path.join(DIR,"plots/stacked_%fsources_pllh.png"%(numsources))
    plot_logProfile_alt([IntC],param_df,like_df,"Stacked",show=False,minlogN=minlogN,maxlogN=maxlogN,save=figname)

