import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def PowerLaw(x,A,pivot,gamma):
    return A * (x/pivot) ** (-gamma)


IX = ['2','27','3']
inds = [2.0,2.7,3.0]
#NUM = ['one','two']
num = 'one'
PIV = [1e9,5e9,10e9]

xxs = np.logspace(np.log10(0.5e9),np.log10(100e9),200)
xarr = np.logspace(np.log10(0.5e9),np.log10(100e9),1000)

c = plt.cm.tab10(np.linspace(0, 1, len(IX))).reshape(len(IX), 4) 
plt.figure(figsize=[10,8],layout='constrained')

for j,ix in enumerate(IX):
    norms = []
    df = pd.read_csv(f"mod-nopiv-stacked-ind{ix}-{num}crab-results.csv")
    pivot = df['pivot']
    indminNorm = df['indminNorm']
    
    norms_cols = [f"norms_{k}" for k in range(200)]
    norms_mat = df[norms_cols].values   # (3 rows, 200 vals)    

    for row in range(norms_mat.shape[0]):
        norms = norms_mat[row,:]
        popt, pcov = curve_fit(PowerLaw,xxs,norms,maxfev=10000)
    
        plt.plot(xxs,norms,color=c[j],label=f"Index = {ix}")
        plt.plot(xarr,PowerLaw(xarr,*popt),color=c[j],
            label=f"Fit Powerlaw; A = {popt[0]}, piv = {popt[1]}, gam = {popt[2]}")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Pivot energy [TeV]")
plt.ylabel("Normalization")
plt.title("Crab calculated spectrum")
plt.legend()
plt.savefig("crab_spec.png")
plt.close()

