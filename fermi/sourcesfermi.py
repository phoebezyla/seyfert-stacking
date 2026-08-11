import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import csv
import os
import matplotlib.pyplot as plt

CATALOG   = "gll_psc_v41.fit"
SOURCES   = "data14-195.csv"
SUMMARY   = "matched_sources.csv"
SED_DIR   = "seds"

summary_rows = []

# Load sources #
df = pd.read_csv("data14-195.csv",sep='\\s+').to_numpy()

sourceName = df[:,0]
RA = df[:,1]
Dec = df[:,2]

coords = SkyCoord(
    ra  = RA * u.deg,
    dec = Dec * u.deg,
)

## Possible column titles ##
#cols = [
# 'Source_Name', 'RAJ2000', 'DEJ2000', 'GLON', 'GLAT', 
# 'Conf_68_SemiMajor', 'Conf_68_SemiMinor', 'Conf_68_PosAng',
# 'Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Conf_95_PosAng',
# 'ROI_num', 'Extended_Source_Name', 'Signif_Avg', 'Pivot_Energy',
# 'Flux1000', 'Unc_Flux1000', 'Energy_Flux100', 'Unc_Energy_Flux100',
# 'SpectrumType', 'PL_Flux_Density', 'Unc_PL_Flux_Density', 'PL_Index',
# 'Unc_PL_Index', 'LP_Flux_Density', 'Unc_LP_Flux_Density', 'LP_Index',
# 'Unc_LP_Index', 'LP_beta', 'Unc_LP_beta', 'LP_SigCurv', 'LP_EPeak',
# 'Unc_LP_EPeak', 'PLEC_Flux_Density', 'Unc_PLEC_Flux_Density',
# 'PLEC_IndexS', 'Unc_PLEC_IndexS', 'PLEC_ExpfactorS', 'Unc_PLEC_ExpfactorS',
# 'PLEC_Exp_Index', 'Unc_PLEC_Exp_Index', 'PLEC_SigCurv', 'PLEC_EPeak',
# 'Unc_PLEC_EPeak', 'Npred', 'Flux_Band', 'Unc_Flux_Band', 'PriorSigma_Band',
# 'nuFnu_Band', 'Sqrt_TS_Band', 'Variability_Index', 'Frac_Variability',
# 'Unc_Frac_Variability', 'Signif_Peak', 'Flux_Peak', 'Unc_Flux_Peak',
# 'Time_Peak', 'Peak_Interval', 'Flux_History', 'Unc_Flux_History',
# 'Sqrt_TS_History', 'ASSOC_FGL', 'ASSOC_FHL', 'ASSOC_GAM1', 'ASSOC_GAM2',
# 'ASSOC_GAM3', 'TEVCAT_FLAG', 'ASSOC_TEV', 'CLASS1', 'CLASS2', 'ASSOC1',
# 'ASSOC2', 'ASSOC_PROB_BAY', 'ASSOC_PROB_LR', 'RA_Counterpart',
# 'DEC_Counterpart', 'Unc_Counterpart', 'Flags']

## Chosen column titles ##
available_cols = [
  'Source_Name','RAJ2000','DEJ2000','Signif_Avg', 'Pivot_Energy',
  'Flux1000', 'Unc_Flux1000', 'Energy_Flux100', 'Unc_Energy_Flux100', 'SpectrumType', 
  'PL_Flux_Density', 'Unc_PL_Flux_Density', 'PL_Index', 'Unc_PL_Index',
  'LP_Flux_Density', 'Unc_LP_Flux_Density', 'LP_Index', 'Unc_LP_Index',
  'LP_beta', 'Unc_LP_beta', 'LP_SigCurv', 'LP_EPeak', 'Unc_LP_EPeak',
  'Signif_Peak', 'Flux_Peak', 'Unc_Flux_Peak',
  'ASSOC_FGL','ASSOC1','ASSOC2','RA_Counterpart','DEC_Counterpart',
  'Unc_Counterpart', 'Flags', 
  ]

array_cols = [
  'Flux_Band', 'Unc_Flux_Band', 'nuFnu_Band', 'Sqrt_TS_Band'
  ]

## Energy band edges? ##
band_edges_MeV = np.array([
    50, 100, 300, 1000, 3000, 
    10000, 30000, 100000, 1000000
    ])

e_lo = band_edges_MeV[:-1]
e_hi = band_edges_MeV[1:]
e_mids = np.sqrt(e_lo * e_hi)  # geometric mean; log-spaced bins
sy_en  = np.array([1.0,5.0,10.0]) * 1e6 # TeV -> MeV
sy_erg = sy_en * 1.602e-6     # MeV -> ergs

## Read FITS file ##
with fits.open(CATALOG) as hdul:
    cat = Table(hdul["LAT_Point_Source_Catalog"].data)
    #hdul.info()
    #print(cat.colnames)

cat_coords = SkyCoord(ra = cat["RAJ2000"], dec = cat["DEJ2000"], unit='deg')

# Match my coordinates to the sources in the catalog #
idx,sep2d,_ = coords.match_to_catalog_sky(cat_coords)

fermiName = cat[idx]["Source_Name"]
assoc     = cat[idx]["ASSOC1"]
fermiRA   = cat[idx]["RAJ2000"]
fermiDec  = cat[idx]["DEJ2000"]

spectype  = cat[idx]["SpectrumType"]
pivot     = cat[idx]["Pivot_Energy"]

flxden_PL     = cat[idx]["PL_Flux_Density"]
flxden_err_PL = cat[idx]["Unc_PL_Flux_Density"]
ix_PL         = cat[idx]["PL_Index"]
ix_err_PL     = cat[idx]["Unc_PL_Index"]

flxden_LP     = cat[idx]["LP_Flux_Density"]
flxden_err_LP = cat[idx]["Unc_LP_Flux_Density"]
ix_LP         = cat[idx]["LP_Index"]
ix_err_LP     = cat[idx]["Unc_LP_Index"]
beta_LP       = cat[idx]["LP_beta"]
beta_err_LP   = cat[idx]["Unc_LP_beta"]

flux_band      = cat[idx]["Flux_Band"]
unc_flux_lower = cat[idx]["Unc_Flux_Band"][:,:,0]
unc_flux_upper = cat[idx]["Unc_Flux_Band"][:,:,1] 
nufnu          = cat[idx]["nuFnu_Band"]

with np.errstate(divide='ignore', invalid='ignore'):
    unc_nufnu_lower = np.where(flux_band > 0,
        nufnu * np.abs(unc_flux_lower) / flux_band,
        np.nan)
    unc_nufnu_upper = np.where(flux_band > 0,
        nufnu * np.abs(unc_flux_upper) / flux_band,
        np.nan)

## Extrapolate to my energies ##
# begin loop over sources #
for i, s in enumerate(sourceName):
    dNdE_PL    = []
    dNdE_LP    = []
    nuFnu_PL   = []
    nuFnu_LP   = []
    uplim_mask = []

    sep_deg = sep2d[i].deg

    print(f"Closest source was {sep_deg:.3f} deg away")
    summary_rows.append(cat[idx[i]])
    # Extrapolate to my energies #
    dNdE_PL = flxden_PL[i] * (sy_en/pivot[i])**ix_PL[i]
    dNdE_LP = flxden_LP[i] * (sy_en/pivot[i])**(-(ix_LP[i] + beta_LP[i] * np.log(sy_en/pivot[i])))

    nuFnu_PL = sy_erg**2 * dNdE_PL/1.602e-6 # erg cm-2 s-1
    nuFnu_LP = sy_erg**2 * dNdE_LP/1.602e-6

    # Separate array for upper limits #
    uplim_mask = np.isnan(unc_flux_lower[i])

    for j in range(len(unc_nufnu_lower[i])):
        if uplim_mask[j]:
            unc_nufnu_lower[i][j] = 0.4 * nufnu[i][j]

    plt.figure(layout='constrained')

    plt.errorbar(e_mids,nufnu[i],
            yerr=[unc_nufnu_lower[i],unc_nufnu_upper[i]],
            uplims=uplim_mask,color='r',fmt='none',
            capsize=10,capthick=1.5,elinewidth=1.5,
            label="Fermi Data")
    for x, y in zip(e_mids, nufnu[i]):
        plt.annotate(f"{x:.0f}", xy=(x, y), xytext=(0, 8),
                 textcoords='offset points',
                 fontsize=8, ha='center')

    plt.scatter(sy_en,nuFnu_PL,label="powerlaw",color='green')
    for x, y in zip(sy_en, nuFnu_PL):
        plt.annotate(f"{x:.0f}", xy=(x, y), xytext=(0, 8),
                 textcoords='offset points',
                 fontsize=8, ha='center')

    plt.scatter(sy_en,nuFnu_LP,label="logParabola",color='blue')
    for x, y in zip(sy_en, nuFnu_LP):
        plt.annotate(f"{x:.0f}", xy=(x, y), xytext=(0, 8),
                 textcoords='offset points',
                 fontsize=8, ha='center')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [MeV]')
    plt.ylabel(r"$\nu F_{\nu}$ [erg cm$^{-2}$ s$^{-1}$]")
    plt.title(f"{s} SED\n{sep_deg} away from {fermiName[i]}")
    plt.legend()
    plt.savefig(f"seds/{s}_sed.png")
    plt.close()
    print(f"{s} SED completed")


## Summary rows ##

## Write csv ##
priority_cols = ["input_name", "input_ra", "input_dec", "sep_deg",
                 "Source_Name", "ASSOC1", "RAJ2000", "DEJ2000",
                 "SpectrumType", "Signif_Avg", "Pivot_Energy"
                 ]

with open(SUMMARY,'w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=priority_cols)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\nDone. Summary written to {SUMMARY}")


