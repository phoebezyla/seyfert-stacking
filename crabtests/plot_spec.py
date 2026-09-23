import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize as opt


def PowerLaw(x,A,pivot,gamma):
    return A * (x/pivot) ** (-gamma)

IX   = ['2','27','3']
inds = [2.0,2.7,3.0]
#NUM = ['one','two']
num  = 'one'

PIV    = [1e9,5e9,10e9]
E_low  = 0.5e9
E_high = 100e9 
xarr   = np.logspace(np.log10(E_low),np.log10(E_high),1000)

c = plt.cm.tab10(np.linspace(0, 1, len(PIV)))
piv_linestyles = ['-', '--', ':']  # one per pivot, reused across indices

## Energy Spectrum Figure ##

for j,ix in enumerate(IX):
    df = pd.read_csv(f"mod-nopiv-stacked-ind{ix}-{num}crab-results.csv")    

    plt.figure(figsize=[10,8],layout='constrained')
    for i, E0 in enumerate(PIV):
        A_best = df.loc[i, 'indminNorm']
        yarr = PowerLaw(xarr, A_best, E0, inds[j])
        plt.plot(xarr,yarr,color=c[i],
            label=f"E0 = {E0:.1e} keV, A = {A_best:.2e}")

        plt.plot(E0, A_best, 'o', color=c[i],label="_nolabel_")

    plt.ylim(1e-27, 5e-17)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Energy [keV]")
    plt.ylabel(r"Flux [??]")
    plt.title(f"Crab Spectrum across pivot assumptions, Index = {inds[j]}")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig(f"crab_spec_ind{ix}.png")
    plt.close()


## Combined Spectra Plot
handles = {}

fig, ax = plt.subplots(figsize=[10, 8], layout='constrained')

for j, ix in enumerate(IX):
    df = pd.read_csv(f"mod-nopiv-stacked-ind{ix}-{num}crab-results.csv")
    for i, E0 in enumerate(PIV):
        A_best = df.loc[i, 'indminNorm']

        yarr = PowerLaw(xarr, A_best, E0, inds[j])
        line, = ax.plot(
            xarr, yarr, color=c[i], ls=piv_linestyles[j],
            label=f"Index = {inds[j]}, E0 = {E0:.1E} keV"
        )
        handles[(i,j)] = line
        ax.plot(E0, A_best, 'o', color=c[i])

ordered_handles = [handles[(i,j)] for i in range(len(PIV)) for j in range(len(IX))]
ordered_labels = [h.get_label() for h in ordered_handles]

ax.set_ylim(1e-27,5e-17)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Energy [keV]")
ax.set_ylabel(r"Flux [keV$^{-1}$ s$^{-1}$ cm$^{-2}$]")
ax.set_title("Spectrum curves: all indices and pivot assumptions")
ax.legend(ordered_handles, ordered_labels, fontsize=8)
ax.grid(True, which='both', alpha=0.3)

fig.savefig("crab_spec_pivot_compare_all.png")
plt.close(fig)


## Likelihood Profile plot ##
norms_cols = [f"norms_{k}"   for k in range(200)]
log_cols   = [f"log_val_{k}" for k in range(200)]
for j, ix in enumerate(IX):
    df = pd.read_csv(f"mod-nopiv-stacked-ind{ix}-{num}crab-results.csv")
    
    bin_c = plt.cm.tab10(np.linspace(0,1,len(df)))
    fig, ax = plt.subplots(figsize=[10,8],layout='constrained')
    
    for i, row in df.iterrows():
        norms = row[norms_cols].values.astype(float)
        logs  = row[log_cols].values.astype(float)
 
        logs_shifted = logs - logs.min()  # minimum is at 0, because we're comparing curves
        # Wilks' theorem: difference between -logL at a given normalization and -logL at the best fit has meaning
        # Difference follows a chi-squared distribution, so eltaLogL = 0.5 --> 68% confidence interval --> red horizontal line
        # wihtout shifting, would need a separate red line for each line to show each line's confidence interval 

        # if I want4ed to see one curve by itseld, I can look at that bin's curve (is minimum well-defined, is curve lopsideed, is there a minimmum within the scanned range?)
        #  If the curve never crossed the threshold (flattens out), then we are in an upper-limit case
        # with one curve, it's easier to compare the profile's minimum against the indminNorm from the joint fit 

        ax.plot(norms, logs_shifted, color=bin_c[i], label=f"E = {PIV[i]:.1E} keV")
        ax.axvline(row['indminNorm'],color=bin_c[i], ls = ":", alpha = 0.6, label="Best-fit normalization")
#        ax.axhline(0.5, color='r', ls = '--', alpha = 0.7, label=r"$\Delta$logL = 0.5")

        ax.set_xlim(1e-25,1e-12)
        ax.set_xscale('log')
        ax.set_xlabel(r"Normalization K []")
        ax.set_ylabel("-logL - min('logL)")
        ax.set_title(f"Likelihood Profiles, Index = {inds[j]}")
        ax.legend()

        fig.savefig(f"crab_likelihood_ind{ix}.png")
        plt.close(fig)
        

## Combined Likelihood Plot ##
fig, ax = plt.subplots(figsize=[10,8],layout='constrained')
handles = {}

for j, ix in enumerate(IX):
    df = pd.read_csv(f"mod-nopiv-stacked-ind{ix}-{num}crab-results.csv")
    for i, row in df.iterrows():
        norms = row[norms_cols].values.astype(float)
        logs  = row[log_cols].values.astype(float)
        logs_shifted = logs - logs.min()  # minimum is at 0, because we're comparing curves
    
        line, = ax.plot(norms, logs_shifted, color=c[i], 
            label=f"E = {PIV[i]:.1E} keV, Ind = {inds[j]:.1E}",
            ls=piv_linestyles[j])
        handles[(i,j)] = line
    
ordered_handles = [handles[(i,j)] for i in range(len(PIV)) for j in range(len(IX))]
ordered_labels = [h.get_label() for h in ordered_handles]

ax.set_xlim(1e-25,1e-12)
ax.set_xscale('log')
ax.set_xlabel(r"Normalization K []")
ax.set_ylabel("-logL - min('logL)")
ax.set_title(f"Likelihood Profiles")
ax.legend(ordered_handles, ordered_labels, fontsize=8)
    
fig.savefig(f"crab_likelihood_combined.png")
plt.close(fig)
