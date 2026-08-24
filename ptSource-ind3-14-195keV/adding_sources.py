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
df = pd.read_csv("dataUnc.csv",sep='\\s+').to_numpy()

numsources = 41
numsourcesstr = '41'
ix = '2'

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

## Begin loop over pivot energies ##
for j, e in enumerate(midE):
    IntC_arr = []
    TS_arr = []
    nullLLH_arr = []

    with open("results-%six-%.1fTeV-individual.csv"%(ix,e), newline="") as f:
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

    ## Begin loop over sources ##
    for i,c in enumerate(sourceName):
        ra = RA[i]
        dec = Dec[i]
        a = A[i]
        
        # Load individual profiles 
        nullLLH = results_df[c]["nullLLH"]
        TSind   = results_df[c]["TS"]
        log_val = results_df[c]["log_val"]
        norms   = results_df[c]["norms"]
        

        # Get likelihood profile around min Norm? #        
        ic = IntervalContainer(lowerE,upperE,norms,log_val,101,
                               weight=A[i],pivot=e*1e9)
        IntC_arr.append(ic)

        nullLLH_arr.append(A[i] * nullLLH)  # need to weight for TS calculations
        TS_arr.append(TSind)

        ## End Source loop ##
        #####################

    ###############################
    ## Stack likelihood profiles ##

    totalnull = np.sum(np.asarray(nullLLH_arr))
    
    # Load source model #
    clm_model_file = "%s/model_files/yml_clm/clm_E_%.1f_TeV_ind%s_modelFile.yml"%(DIR,e,ix)
    clm = threeML.load_model(clm_model_file)

    print(clm.finalNorm.spectrum.main.Powerlaw.K.value)

    cl = CastroLike("stacked",IntC_arr)
    cl.set_model(clm)
    data = DataList(cl)
    fjl = JointLikelihood(clm,data,verbose=False)
    #fjl.set_minimizer("ROOT")

    param_df, like_df = fjl.fit(quiet=True)

    fig = cl.plot()
    fig.savefig("%s/plots/modcastrolike-%d.png"%(DIR,e))
    
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

    # Plot stacked profile #    
    figname = os.path.join(DIR,"plots/modstacked_%.1fTeV_pllh.png"%(e))
    plot_logProfile_alt([IntC],param_df,like_df,"Stacked",show=False,minlogN=minlogN,maxlogN=maxlogN,save=figname)


    ## Build results dataframe ##
    results[e] = {
        "indminNorm": param_df['value'][0],
        "TS"        : TS_stacked,
#        "loglike"   : loglike,
#        "credInt"   : credInt,
#        "resHigh"   : resHigh,
        "norms"     : norms_Stack,
        "log_val"   : log_val_Stack,
        }

## Save full results dictionary to one csv per index ##
f = open("%s/results-stacked-ind%s-mod.csv"%(DIR,ix),'w',newline='')
writer = csv.writer(f)
writer.writerow(['pivot','indminNorm','TS'] +\
#,'loglike'] +\
    [f"norms_{i}" for i in range(200)] +\
    [f"log_val_{i}" for i in range(200)])

for pivot,vals in results.items():
    writer.writerow([pivot, vals['indminNorm'], vals['TS']]\
        #vals['resBay'], vals['resLow'],vals['resHigh']] 
        + list(vals['norms']) + list(vals['log_val']))
f.close()
