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
    e_mids     = sed[:,3]
    flux       = sed[:,4] # photon cm2 s
    fluxerr_lo = sed[:,5]
    fluxerr_hi = sed[:,6]
    nuFnu      = sed[:,7] # erg cm2 s
    nuerr_lo   = sed[:,8]
    nuerr_hi   = sed[:,9]

    e_err  = e_maxs - e_mids    

    # Separate array for upper limits #
    uplim_mask = np.isnan(fluxerr_lo)
    print(uplim_mask)
    for j in range(len(nuerr_lo)):
        if uplim_mask[j]:
            nuerr_lo[j] = 0.4 * nuFnu[j]

    plt.figure(layout='constrained')
    plt.errorbar(e_mids,nuFnu,
            yerr=[nuerr_lo,nuerr_hi],
            uplims=uplim_mask,fmt='o',color='r',
            capsize=4,capthick=1.5,elinewidth=1.5)
            
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [MeV]')
    plt.ylabel(r"$\nu F_{\nu}$ [erg cm$^{-2}$ s$^{-1}$]")
    plt.title(f"{s} SED\n{sep_deg[i]} away from {fermiName[i]}")
    plt.savefig(f"seds/{s}_sed.png")
    plt.close()
    print(f"{s} SED completed")
