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
match_rad = 1.0
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

cat_coords = SkyCoord(ra = cat["RAJ2000"], dec = cat["DEJ2000"], unit='deg')

# Match my coordinates to the sources in the catalog #
idx,sep2d,_ = coords.match_to_catalog_sky(cat_coords)

## Check column names ##
candidate_cols = [
    "Source_Name","RAJ2000","DEJ2000","GLON","GLAT",
    "SpectrumType","Pivot_Energy",
    "PL_Flux_Density","PL_Index",
    "LP_Flux_Density","LP_Index","LP_beta",
    "PLEC_Flux_Density","PLEX_IndexS","PLEX_ExpfactorS","PLEC_Exp_Index",
    "Flux1000","Unc_Flux1000",
    "Energy_Flux100","Unc_Energy_Flux100",
    "Signif_Avg","Variability_Index",
    "ASSOC1","CLASS1",
]
available_cols = [c for c in candidate_cols if c in cat.colnames]
missing = set(candidate_cols) - set(available_cols)
if missing:
    print(f"Note: These columns weren't found in this catalog: {missing}")

# Band-resolved arrays for SEDs #
band_flux_col = "Flux_Band" if "Flux_Band" in cat.colnames else None
band_flux_unc_col = 'Unc_Flux_Band' if "Unc_Flux_Band" in cat.colnames else None
band_nufnu_col = "nuFnu_Band" if "nuFnu_band" in cat.colnames else None

# Energy band edges? #
energy_bounds = None
with fits.open(CATALOG) as hdul:
    for h in hdul:
        if h.name.lower() in ('energybounds','energy_bounds'):
            energy_bounds = Table(h.data)
            break

# summary rows #
summary_rows = []
for i, s in enumerate(sourceName):
    sep_deg = sep2d[i].deg
    matched = sep_deg <= match_rad
    row = {
        'input_name': s,
        'input_ra'  : RA[i],
        'input_dec' : Dec[i],
        'matched'   : matched,
        'sep_deg'   : round(sep_deg, 5),
    }

    if matched: 
        cat_row = cat[idx[i]]
        for col in available_cols:
            row[col] = cat_row[col]

        # if band columns exist, save source SED
        if band_flux_col and band_nufnu_col:
            sed_path = os.path.join(SED_DIR, f"{s}_sed.csv")
            flux_band = np.array(cat_row[band_flux_col])
            nufnu_band = np.array(cat_row[band_nufnu_col])
            flux_unc = (np.array(cat_row[band_flux_unc_col]) if band_flux_unc_col else None)

            with open(sed_path,'w',newline='') as f:
                writer = csv.writer(f)
                header = ["band_index","flux_ph_cm2_s","nuFnu_erg_cm2_s"]
                if energy_bounds is not None:
                    header = ["band_index","e_min_MeV",
                              'e_max_MeV','flux_ph_cm2_s',
                              'nuFnu_erg_cm2_s']
                writer.writerow(header)
                for b in range(len(flux_band)):
                    if energy_bounds is not None:
                        writer.writerow([
                            b,
                            energy_bounds["LowerEnergy"][b],
                            energy_bounds['UpperEnergy'][b],
                            flux_band[b],
                            nufnu_band[b],
                        ])
                    else:
                        writer.writerow([b,flux_band[b],nufnu_band[b]])
    else:
        print(f"No match within {match_rad} for {s}")
        print(f"Closest source was {sep_deg:.3f} deg away")
    
    summary_rows.append(row)

## Write csv ##
priority_cols = ["input_name", "input_ra", "input_dec", "matched", "sep_deg",
                  "Source_Name", "ASSOC1", "RAJ2000", "DEJ2000"]

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
          

