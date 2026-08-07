import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load source CSV #
df = pd.read_csv("matched_sources.csv",sep=',').to_numpy()

inputName = df[:,0]
fermiName = df[:,4]
sep_deg   = df[:,3]

# begin loop over sources #
for i, s in enumerate(inputName):
    sed = pd.read_csv(f"seds/{s}_sed.csv",sep=',').to_numpy()
    e_mins     = sed[:,1] # MeV
    e_maxs     = sed[:,2] # MeV
    flux       = sed[:,3] # photon cm2 s
    fluxerr_lo = sed[:,4]
    fluxerr_hi = sed[:,5]
    nuFnu      = sed[:,6] # erg cm2 s
    nuerr_lo   = sed[:,7]
    nuerr_hi   = sed[:,8]

    e_mids = (e_mins + e_maxs)/2
    e_err  = e_maxs - e_mids    

    # Separate array for upper limits #
    uplim_mask = np.isnan(fluxerr_lo)

    plt.figure(layout='constrained')
    plt.errorbar(e_mids,nuFnu,
            xerr=e_err,yerr=[nuerr_lo,nuerr_hi],
            uplims=uplim_mask,fmt='o')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [MeV]')
    plt.ylabel(r"$\nu F_{\nu}$ [erg cm$^{-2}$ s$^{-1}$]")
    plt.title(f"{s} SED\n{sep_deg[i]} away from {fermiName[i]}")
    plt.savefig(f"seds/{s}_sed.png")
    plt.close()
    print(f"{s} SED completed")
