#import matplotlib as mpl
#mpl.use("Agg")
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
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from threeML import *
    from threeML.plugins.experimental.CastroLike import *
    from hawc_hal import HAL, HealpixConeROI, HealpixMapROI

def plot_logProfile(IntC,param_df,like_df,minlogN=-23.,maxlogN=-9.,show=False,save=None):
    finalnorm = np.linspace(minlogN,maxlogN,200)
    finalnorm = np.power(10,finalnorm)
    totalllh = np.zeros(200)
    #llh1 = np.zeros(200)
    #llh2 = np.zeros(200)
    for i,fn in enumerate(finalnorm):
        for j,cont in enumerate(IntC):
            totalllh[i] += cont(fn)
    llhinterp = InterpolatedUnivariateSpline(np.log10(finalnorm),totalllh,k=1,ext=0)
    minNorm = param_df['value'][0]#np.power(10,res.x)
    minLLH = like_df.iloc[1]['-log(likelihood)']
    fig,sbu = plt.subplots()
    plt.plot(finalnorm,-totalllh-minLLH,'b',markersize=2) #totallh needs a min, to make it positive
    plt.vlines(minNorm,0.,3.0,linestyles='--')
    plt.xscale('log')
    plt.ylim(0,2.71)
    plt.xlim(np.power(10,minlogN),np.power(10,maxlogN))
    plt.xlabel("Normalization [kev-1 s-1 cm-2]")
    plt.ylabel("LLH-LLHmin")
    plt.grid()
    if show:
        plt.show()
    if save is not None:
        fig.savefig("{}".format(save))
    fig.clear()
    plt.close(fig)


start = time.perf_counter()
print("Starting time: ",start)

numSources = 6 # 1 to 51

DATADIR = '/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/'
DIR = '/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/timing/'
MAP = os.path.join(DATADIR,'maptree-fhit2pct-pass5f-mlp-chunk1-1510.root')
DR = os.path.join(DATADIR, 'detRes-fhit2pct-pass5f-mlp-refit.root')

print(MAP)
print(DR)

lowerE_long = np.logspace(np.log10(0.5),np.log10(10),4)[0:3]  # three vals 500 GeV to 10 TeV
lowerE = [lowerE_long[0]]
print(lowerE)

df = pd.read_csv("data.csv",sep='\\s+').to_numpy()

sourceName = df[:numSources,0]
RA = df[:,1]
Dec = df[:,2]
A = df[:,3]

print("The total weights are", A)

Atotal = 0
for c in range(len(sourceName)):
    Atotal += A[c]
IntC = []
nullLLH = [] 
bestGuess = 1e-26

results = []
resultsLow = [] 
resultsHigh = [] 
TSArray = []

sourceStart = np.zeros(len(sourceName))
sourceEnd = np.zeros(len(sourceName))

for e in lowerE:
    IntC[:] = []
    nullLLH[:] = []
    lowe = e    # lowerE[0]
    uppe = np.power(10,np.log10(lowe)+0.25)
    mide = np.power(10,np.log10(lowe)+0.125)
    print('{}'.format("#####"*4))
    print('Energy: {} TeV'.format(mide))
    
    outfileName = os.path.join(DIR,"data_stacking_E%f.txt"%(numSources))
    print("Datafile is ",outfileName)

    for i,c in enumerate(sourceName):
        sourceStart[i] = time.perf_counter()
        print("Source start time: ",sourceStart[i])

        ra = RA[i]
        dec = Dec[i]
        data_radius = 5.
        model_radius = 8.
        print(c, ra, dec)

        spectrum = Powerlaw() #*Constant()  #*StepFunction() for dec-dependednt bins
        spectrum.index = -3.0
        spectrum.index.fix=True
        spectrum.K.unit = (u.keV * u.s * u.cm**2 )**(-1)
        spectrum.K = 1e-21 * A[i]
        spectrum.K.min_value = 1e-29
        spectrum.K.max_value = 1e-3
        spectrum.K.fix = False
        spectrum.piv = mide
        spectrum.piv.fix = True
        spectrum.piv.unit = u.TeV

        source = PointSource(c,ra,dec,spectrum)

        lm = Model(source)

        roi = HealpixConeROI(data_radius=data_radius,
                         model_radius=model_radius,
                         ra=ra,
                         dec=dec)
        llh  = HAL("Likelihood_{}".format(c),MAP,DR,roi)
       
        bins = ['B2C0','B2C1','B3C0','B3C1','B4C0','B4C1','B5C0','B5C1','B6C0','B6C1','B7C0','B7C1','B8C0','B8C1','B9C0','B9C1','B10C0','B10C1']
        print(bins)
        llh.set_active_measurements(bin_list=bins)
        datalist = DataList(llh)

        poissonFluctuate = False
        if poissonFluctuate is True:
            print("Fluctuating bkg and using as data")
            ejl = JointLikelihood(Model(),datalist,verbose=False)
            poiss_data = []
            for data in ejl.data_list.values():
                new_data = data.get_simulated_dataset("%s_sim" % data.name)
                poiss_data.append(new_data)
            datalist = DataList(*poiss_data)

        print("{}".format("*"*20))
        print("Datalist made for pulsar {}".format(c))
        jl = JointLikelihood(lm,datalist,verbose=False)
        print("Fitting time!!")
        start=time.time()
        jl.set_minimizer("ROOT")
        param_df, like_df = jl.fit(quiet=True)
        stop =time.time()
        print("Time to make fit: ",stop-start, "sec")

        print("{}".format("*"*15))
        print("Parameter Results: ")
        print(param_df)
        print("{}".format("*"*15))
        print("Likelihood Results: ")
        print(like_df)

        indminNorm = param_df['value'][0] # in kev-1 s-1 cm-2
        if indminNorm < bestGuess:
            bestGuess = indminNorm
        print("Getting Likelihood profile around minimum Norm")
        norms = np.linspace(np.log10(indminNorm)-5,np.log10(indminNorm)+5,200)
        log_val = np.zeros(200)
        for j in range(200):
            jl.verbose=False
            log_val[j] = jl.minus_log_like_profile(norms[j]) # need logllh, not -logllh
        norms = np.power(10,norms)
        #np.savez("ROI_{}.npz".format(i+1),norms,log_val)
        IntC.append(IntervalContainer(i+1,i+2,norms,-log_val,101))
        a = jl.compute_TS(c,like_df)
        nullLLH.append(a.iloc[0]['Null hyp.'])
        print("TS: {}".format(a.iloc[0]['TS']))
        print("{}\n".format("*"*20))
        print("Individual logProfile")
        
        #figname = os.path.join(DIR,"plots/E{:0.2f}/{}_E{:0.2f}_pllh.png".format(mide,c,mide))
        #plot_logProfile([IntC[-1]],param_df,like_df,minlogN=np.log10(indminNorm)-3,maxlogN=np.log10(indminNorm)+1.5,show=False,save=figname)
        
        sourceEnd[i] = time.perf_counter()
        print("Time elapsed is",sourceEnd[i]-sourceStart[i])

        with open(outfileName,'a') as datafile:
            datafile.write(str(c)+" TS: ")
            datafile.write(str(a.iloc[0])+'\n')
            datafile.write("Parameter Results: "+str(param_df)+'\n')
            datafile.write("Likelihood Results: "+str(like_df)+'\n')
            datafile.write("Time elapsed: ")
            datafile.write(str(sourceEnd[i]-sourceStart[i])+'\n'+'\n')

    totalnull = np.sum(np.asarray(nullLLH))
    print("Stacking Likelihoods")
    print("Total factor A: {}".format(Atotal))
    print("Total nullLLH: {}".format(totalnull))
    cl = CastroLike("stacked",IntC)
    newdata = DataList(cl)
    
    normalization = Powerlaw()
    normalization.K = 1e-21
    normalization.K.min_value = 1e-30
    normalization.K.max_value = 1e-3
    normalization.K.unit = (u.keV * u.s * u.cm**2 )**(-1)
    normalization.K.free = True
    normalization.index = -3.0
    normalization.index.free = False
    fsource = PointSource("finalNorm",ra=0,dec=0,spectral_shape=normalization)
    clm = Model(fsource)

    print(clm)

    fjl = JointLikelihood(clm,newdata,verbose=True)

    print("Fitting time!!")
    start=time.time()
    #fjl.set_minimizer("ROOT")
    param_df, like_df = fjl.fit(quiet=False)
    stop =time.time()
    print("Time to make fit: ",stop-start, "sec")

    print("{}".format("*"*15))
    print("Parameter Results: ")
    print(param_df)
    print("{}".format("*"*15))
    print("Likelihood Results: ")
    print(like_df)


    TS =  2 * (totalnull - like_df.iloc[1]['-log(likelihood)'])
    TSArray.append(TS)
    print("Total TS: {}\n".format(TS))
 
    print(TSArray)

    ####### Check final loglikelihood profile
    indminNorm = param_df['value'][0]
    if indminNorm<normalization.K.min_value:
        normalization.k.min_value = 1e-40    
    
    #figname = os.path.join(DIR,"plots/Stacked_E{:0.2f}_pllh.png".format(mide))
    #plot_logProfile(IntC,param_df,like_df,show=False,minlogN=np.log10(indminNorm)-3,maxlogN=np.log10(indminNorm)+1.5,save=figname)
    end = time.perf_counter()
    print("End time: ",end)

    with open(outfileName,'a') as datafile:
        datafile.write("\nTotal time elapsed: ")
        datafile.write(str(end-start)+'\n')
        datafile.write("Total param results: "+str(param_df)+'\n')
        datafile.write("Total Likelihood results: "+str(like_df)+'\n')
        datafile.write("Total TS: ")
        datafile.write(str(TS)+'\n'+'\n')

