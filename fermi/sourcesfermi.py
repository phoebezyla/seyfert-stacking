import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import csv
import os

CATALOG   = "gll_psc_v41.fit"
SOURCES   = "data14-195.csv"
SUMMARY   = "matched_sources.csv"
SED_DIR   = "seds"

# Load sources #
df = pd.read_csv("data14-195.csv",sep='\\s+').to_numpy()

sourceName = df[:,0]
RA = df[:,1]
Dec = df[:,2]

coords = SkyCoord(
    ra  = RA * u.deg,
    dec = Dec * u.deg,
)

with fits.open(CATALOG) as hdul:
    hdul.info()
    cat = Table(hdul["LAT_Point_Source_Catalog"].data)
    print(cat.colnames)

cat_coords = SkyCoord(ra = cat["RAJ2000"], dec = cat["DEJ2000"], unit='deg')

# Match my coordinates to the sources in the catalog #
idx,sep2d,_ = coords.match_to_catalog_sky(cat_coords)

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
  'Flux1000', 'Unc_Flux1000', 'Energy_Flux100', 'Unc_Energy_Flux100',
  'SpectrumType', 'PL_Flux_Density', 'Unc_PL_Flux_Density', 'PL_Index',
  'Unc_PL_Index', 'LP_Flux_Density', 'Unc_LP_Flux_Density', 'LP_Index',
  'Unc_LP_Index', 'LP_beta', 'Unc_LP_beta', 'LP_SigCurv', 'LP_EPeak',
  'Unc_LP_EPeak', 'Sqrt_TS_Band','Signif_Peak', 'Flux_Peak', 'Unc_Flux_Peak',
  'ASSOC_FGL','ASSOC1','ASSOC2','RA_Counterpart','DEC_Counterpart',
  'Unc_Counterpart', 'Flags', 'Flux_Band', 'Unc_Flux_Band','nuFnu_Band'
  ]


# Energy band edges? #
band_edges_MeV = np.array([50, 100, 300, 1000, 3000, 10000, 30000, 100000, 1000000])
e_lo = band_edges_MeV[:-1]
e_hi = band_edges_MeV[1:]
e_mids = np.sqrt(e_lo * e_hi)  # geometric mean; log-spaced bins

# summary rows #
summary_rows = []
for i, s in enumerate(sourceName):
    sep_deg = sep2d[i].deg
    row = {
        'input_name': s,
        'input_ra'  : RA[i],
        'input_dec' : Dec[i],
        'sep_deg'   : round(sep_deg, 5),
    }

    cat_row = cat[idx[i]]
    for col in available_cols:
        row[col] = cat_row[col]

    # save source SED #
    sed_path   = os.path.join(SED_DIR, f"{s}_sed.csv")
    flux_band  = np.array(cat_row["Flux_Band"])
    nufnu_band = np.array(cat_row["nuFnu_Band"])
    flux_unc   = np.array(cat_row["Unc_Flux_Band"])

    unc_flux_lower = cat_row["Unc_Flux_Band"][:, 0]
    unc_flux_upper = cat_row["Unc_Flux_Band"][:, 1]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unc_nufnu_lower = np.where(flux_band > 0,
                                nufnu_band * np.abs(unc_flux_lower) / flux_band,
                                np.nan)
        unc_nufnu_upper = np.where(flux_band > 0,
                                nufnu_band * np.abs(unc_flux_upper) / flux_band,
                                np.nan)

    with open(sed_path,'w',newline='') as f:
        writer = csv.writer(f)
        header = ["band_index","e_min_MeV",'e_max_MeV','e_mid_MeV',
                      'flux_ph_cm2_s','fluxerr_lo','fluxerr_hi',
                      'nuFnu_erg_cm2_s','nuFnuerr_lo','nuFnuerr_hi']
        writer.writerow(header)
        for b in range(len(flux_band)):
            writer.writerow([
                    b,
                    e_lo[b],
                    e_hi[b],
                    e_mids[b],
                    flux_band[b],
                    unc_flux_lower[b],
                    unc_flux_upper[b],
                    nufnu_band[b],
                    unc_nufnu_lower[b],
                    unc_nufnu_upper[b],
            ])

    print(f"Closest source was {sep_deg:.3f} deg away")
    summary_rows.append(row)

## Write csv ##
priority_cols = ["input_name", "input_ra", "input_dec", "sep_deg",
                 "Source_Name", "ASSOC1", "RAJ2000", "DEJ2000",
                 "SpectrumType",
                 ]

all_fields = set()
for r in summary_rows:
    all_fields.update(r.keys())

remaining = sorted(all_fields - set(priority_cols))
fieldnames = priority_cols + remaining

with open(SUMMARY,'w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\nDone. Summary written to {SUMMARY}")
print(f"Per-source SEDs (those available, at least) written to {SED_DIR}/")


