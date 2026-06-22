#!/bin/sh
#SBATCH --time=30:00:00 --mem-per-cpu=8000mb
#
# RUN:
# __RUN__

#source /data/disk01/home/zylaphoe/micromamba/etc/profile.d/mamba.sh
#mamba activate new_hal


## Set paths ##
DATADIR='/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert'

MAP="${DATADIR}/maptree-fhit2pct-pass5f-mlp-chunk1-1510.root"
DR="${DATADIR}/detRes-fhit2pct-pass5f-mlp-refit.root"

CSV='data14-195.csv'

source /data/disk01/home/zylaphoe/hawc_software/init_aerie.sh 

## AERIE -- combine maps and plot
python <<EOF 
import os, sys
import numpy as np

DATADIR = '${DATADIR}'
DIR = DATADIR + '/ptSource-ind3-14-195keV/model_files'
MAPDIR = DIR + '/hd5_files'
FITSDIR = DIR + '/fits_files'
PNGDIR = DIR + '/png_files'
CSV = '${CSV}'


df = np.genfromtxt(CSV,dtype=None,encoding=None,names=True)
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
    filelist = [
        "%s/%s_bin%s.fits.gz"%(FITSDIR,source,binNo)
        for binNo in bins
    ]
    print('List of files: %s')%(filelist)
    
    combo = '%s/%s_comb'%(FITSDIR,source)
    print('Combined file: %s'%(combo))

    ret = os.system(
        "aerie-apps-combine-maps --inputs %s -o %s"%(' '.join(filelist),combo)
    )

print("All combined FITS files created")

EOF
