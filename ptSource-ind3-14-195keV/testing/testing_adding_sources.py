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

import threeML.plugins.experimental.CastroLike as cl_module
print(cl_module.__file__)

## Set paths ##
DATADIR = '/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/'
DIR = '/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/seyfert-stacking-git/ptSource-ind3-14-195keV'
MAP = os.path.join(DATADIR,'maptree-fhit2pct-pass5f-mlp-chunk1-1510.root')
DR = os.path.join(DATADIR, 'detRes-fhit2pct-pass5f-mlp-refit.root')

## Define energy bins ##
midE = np.array([1.0,5.0,10.0]) # TeV
lowerE = 0.5 * 1e9              # keV 
upperE = 100 * 1e9              # keV

## Load CSV and initialize arrays ##
df = pd.read_csv("data14-195.csv",sep='\\s+').to_numpy()

#numsources = 41
#numsourcesstr = '41'
ix = '3'

sourceName = df[:,0]
RA = df[:,1]
Dec = df[:,2]
A = df[:,3]

Atotal = 0
for c in range(len(sourceName)):
    Atotal += A[c]

data_radius = 5.
model_radius = 8.

results_df = {}
results = {}


bins = ['B2C0','B2C1','B3C0','B3C1','B4C0','B4C1','B5C0','B5C1',
                 'B6C0','B6C1','B7C0','B7C1','B8C0',
                 'B8C1','B9C0','B9C1','B10C0','B10C1']

## Begin loop over sources ##
for i, c in enumerate(sourceName):
    ra = RA[i]
    dec = Dec[i]
    a = A[i]

    results = {}
   
    for j,e in enumerate(midE):
        IntC_arr = []
        nullLLH_arr = []
        TS_arr = []
 
        with open("results-%six-%.1fTeV-individual.csv"%(ix,e), newline="") as f:
            reader = csv.DictReader(f)
            row_found = None
            for row in reader:
                if row["sourceName"] == c:
                    row_found = row
                    break
        if row_found is None:
            raise ValueError(f"source {c} is not found")

        nullLLH = float(row_found["nullLLH"])
        TSind   = float(row_found["TS"])
        log_val = [float(row_found[f"log_val_{k}"]) for k in range(200)]
        norms   = [float(row_found[f"norms_{k}"])   for k in range(200)]


        # Get likelihood profile around min Norm? #        
        ic = IntervalContainer(lowerE,upperE,norms,log_val,101,
                               weight=a,pivot=e*1e9)
        IntC_arr.append(ic)
        nullLLH_arr.append(A[i] * nullLLH)  # need to weight for TS calculations
        TS_arr.append(TSind)

        totalnull = np.sum(np.asarray(nullLLH_arr))
    
        # Load source model #
        clm_model_file = "%s/model_files/yml_clm/clm_E_%.1f_TeV_ind%s_modelFile.yml"%(DIR,e,ix)
        clm = threeML.load_model(clm_model_file)
        print(clm.finalNorm.spectrum.main.Powerlaw.K.value)
    
        cl = CastroLike("stacked",IntC_arr)
        cl.set_model(clm)
        data = DataList(cl)
        fjl = JointLikelihood(clm,data,verbose=False)
        fjl.set_minimizer("ROOT")
    
        param_df, like_df = fjl.fit(quiet=True)
    
        
        print("Parameter results: %s"%(param_df))
        print("Likelihood results: %s"%(like_df))
    
        # Built total TS array #
        TS_stacked =  2 * (totalnull - like_df.iloc[1]['-log(likelihood)'])
        print("Total TS: %s"%(TS_stacked))
        
    
        ####### Check final loglikelihood profile
        indminNorm = param_df['value'][0]
        minlogN=np.log10(indminNorm) - 1.5
        maxlogN=np.log10(indminNorm) + 1.5
       
        norms_Stack,log_val_Stack,a_Stack = StackingAnalysis.likelihood_profile(indminNorm,fjl,param_df,like_df,"Stacked",computeTS=False)
     
        IntC = IntervalContainer(lowerE,upperE,norms_Stack,log_val_Stack,101)
    
    
        ## Build results dataframe ##
        results[e] = {
            "indminNorm": param_df['value'][0],
            "TS"        : TS_stacked,
            "norms"     : norms_Stack,
            "log_val"   : log_val_Stack,
            }

    ## Save full results dictionary to one csv per index ##
    f = open("%s/sourcebysource/results-stacked-ind%s-%s.csv"%(DIR,ix,c),'w',newline='')
    writer = csv.writer(f)
    writer.writerow(['pivot','indminNorm','TS'])
    
    for pivot,vals in results.items():
        writer.writerow([pivot, vals['indminNorm'], vals['TS']])
    f.close()
