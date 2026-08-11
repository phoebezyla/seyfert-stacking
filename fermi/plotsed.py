import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

band_edges_MeV = np.array([50,100,300,1000,3000,10000,30000,100000,1000000])
e_mins = band_edges_MeV[:-1]
e_maxs = band_edges_MeV[1:]
e_mids = np.sqrt(e_mins * e_maxs) #geometric mean
e_err  = e_maxs - e_mids

sy_en  = np.array([1.0,5.0,10.0]) * 1e6 # TeV -> MeV
sy_erg = sy_en * 1.602e-6     # MeV -> ergs

# Load source CSV #
df = pd.read_csv("matched_sources.csv",sep=',')

inputName = df["input_name"]
fermiName = df["Source_Name"]
sep_deg   = df["sep_deg"]

flux       = df["Flux_Band"]
fluxerr_lo = df["Unc_Flux_Band"][:,0]
fluxerr_hi = df["Unc_Flux_Band"][:,1]


nuFnu      = df["nuFnu_Band"]
with np.errstate(divide='ignore', invalid='ignore'):
    unc_nufnu_lower = np.where(flux_band > 0,
        nufnu_band * np.abs(unc_flux_lower) / flux_band,
        np.nan)
    unc_nufnu_upper = np.where(flux_band > 0,
        nufnu_band * np.abs(unc_flux_upper) / flux_band,
        np.nan)


spec_type = df["SpectrumType"]
pivot     = df["Pivot_Energy"]

ix_PL     = df["PL_Index"]
flxden_PL = df["PL_Flux_Density"]

ix_LP     = df["LP_Index"]
flxden_LP = df["LP_Flux_Density"]
beta_LP   = df["LP_beta"]


# begin loop over sources #
for i, s in enumerate(inputName):
    dNdE_PL    = []
    dNdE_LP    = []
    nuFnu_PL   = []
    nuFnu_LP   = []
    uplim_mask = []

    # Extrapolate to my energies #
    dNdE_PL = flxden_PL[i] * (sy_en/pivot[i])**ix_PL[i] 
    dNdE_LP = flxden_LP[i] * (sy_en/pivot[i])**(-(ix_LP[i] + beta_LP[i] * np.log(sy_en/pivot[i])))

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
            capsize=10,capthick=1.5,elinewidth=1.5,
            label="Fermi Data")
    for x, y in zip(e_mids, nuFnu):
        plt.annotate(f"{x:.0f}", xy=(x, y), xytext=(0, 8),
                 textcoords='offset points', 
                 fontsize=8, ha='center')

    plt.scatter(sy_erg,nuFnu_PL,label="powerlaw",color='green')
    for x, y in zip(sy_erg, nuFnu_PL):
        plt.annotate(f"{x:.0f}", xy=(x, y), xytext=(0, 8),
                 textcoords='offset points',
                 fontsize=8, ha='center')
    
    plt.scatter(sy_erg,nuFnu_LP,label="logPara",color='blue')
    for x, y in zip(sy_erg, nuFnu_LP):
        plt.annotate(f"{x:.0f}", xy=(x, y), xytext=(0, 8),
                 textcoords='offset points',
                 fontsize=8, ha='center')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [MeV]')
    plt.ylabel(r"$\nu F_{\nu}$ [erg cm$^{-2}$ s$^{-1}$]")
    plt.title(f"{s} SED\n{sep_deg[i]} away from {fermiName[i]}")
    plt.legend()
    plt.savefig(f"seds/{s}_sed.png")
    plt.close()
    print(f"{s} SED completed")
