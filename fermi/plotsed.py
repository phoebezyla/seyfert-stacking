import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sy_en  = [1.0 5.0 10.0] * 1e6 # TeV -> MeV
sy_erg = sy_en * 1.602e-6     # MeV -> ergs

# Load source CSV #
df = pd.read_csv("matched_sources.csv",sep=',')

inputName = df["input_name"]
fermiName = df["Source_Name"]
sep_deg   = df["sep_deg"]
spec_type = df["SpectrumType"]
pivot     = df["Pivot_Energy"]
ix_PL     = df["PL_Index"]
flxden_PL = df["PL_Flux_Density"]
ix_LP     = df["LP_Index"]
flxden_LP = df["LP_Flux_Density"]
beta_LP   = df["LP_beta"]
ix_EC     = df["PLEC_Exp_Index"]
flxden_EC = df["PLEC_Flux_Density"]


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

    # Extrapolate to my energies #
    dNdE_PL = flxden_PL * (sy_en/pivot)**ix_PL 
    dNdE_LP = flxden_LP * (sy_en/pivot)**(-(ix_LP + beta_LP * np.log(sy_en/pivot)))

    nuFnu_PL = sy_erg**2 * dNdE_PL/1.602e-6 # erg cm-2 s-1
    nuFnu_LP = sy_erg**2 * dNdE_LP/1.602e-6 

    # Separate array for upper limits #
    uplim_mask = np.isnan(fluxerr_lo)
    print(uplim_mask)
    for j in range(len(nuerr_lo)):
        if uplim_mask[j]:
            nuerr_lo[j] = 0.4 * nuFnu[j]

    plt.figure(layout='constrained')
    plt.errorbar(e_mids,nuFnu,
            yerr=[nuerr_lo,nuerr_hi],
            uplims=uplim_mask,color='r',fmt='none',
            capsize=10,capthick=1.5,elinewidth=1.5)
            
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [MeV]')
    plt.ylabel(r"$\nu F_{\nu}$ [erg cm$^{-2}$ s$^{-1}$]")
    plt.title(f"{s} SED\n{sep_deg[i]} away from {fermiName[i]}")
    plt.savefig(f"seds/{s}_sed.png")
    plt.close()
    print(f"{s} SED completed")
