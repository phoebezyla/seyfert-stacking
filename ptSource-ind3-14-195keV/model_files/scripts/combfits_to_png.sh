#!/bin/sh
#SBATCH --time=30:00:00 --mem-per-cpu=8000mb
#
# RUN:
# __RUN__


## Set paths ##
DATADIR='/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert'

MAP="${DATADIR}/maptree-fhit2pct-pass5f-mlp-chunk1-1510.root"
DR="${DATADIR}/detRes-fhit2pct-pass5f-mlp-refit.root"

CSV='data14-195.csv'

export DATADIR CSV

source /data/disk01/home/zylaphoe/hawc_software/init_aerie.sh 


## AERIE -- plot
python <<'EOF' 
import os, sys
import numpy as np

DATADIR = os.environ['DATADIR']
DIR = "%s/ptSource-ind3-14-195keV/model_files"%(DATADIR)
MAPDIR = '%s/hd5_files'%(DIR)
FITSDIR = '%s/fits_files'%(DIR)
PNGDIR = '%s/png_files'%(DIR)
CSV = os.environ['CSV']


df = np.genfromtxt(CSV, dtype=None, encoding='utf-8',names=True)
sourceName = df['Source']
RA = df['RA']
Dec = df['Dec']

data_radius = 5.

bins = [
    'B2C0','B2C1','B3C0','B3C1','B4C0','B4C1',
    'B5C0','B5C1','B6C0','B6C1','B7C0','B7C1',
    'B8C0','B8C1','B9C0','B9C1','B10C0','B10C1'
]  

for i, source in enumerate(sourceName):
    ra = RA[i]
    dec = Dec[i]

    combo ='%s/%s_comb.fits.gz'%(FITSDIR,source)
    print("Combined file: %s"%(combo))
    
    output_png = '%s/%s.png'%(PNGDIR,source)
    print('Output file: %s'%(output_png))

    ret = os.system(
        "plotMercator.py %s --origin %f %f %f %f -o %s"%(combo,ra,dec,data_radius,data_radius,output_png)
    )

print("All PNGs created")

EOF
