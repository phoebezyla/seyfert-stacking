#!/bin/sh
#SBATCH --time=30:00:00 --mem-per-cpu=8000mb
#
# RUN:
# __RUN__

source /data/disk01/home/zylaphoe/micromamba/etc/profile.d/mamba.sh
mamba activate new_hal


## Set paths ##
DATADIR='/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/'

MAP='${DATADIR}/maptree-fhit2pct-pass5f-mlp-chunk1-1510.root'
DR='${DATADIR}/detRes-fhit2pct-pass5f-mlp-refit.root'

CSV='data14-195.csv'

## HAL -- generate FITS files
python <<EOF
import os, sys, time
import pandas as pd

DATADIR = '${DATADIR}'
DIR = f'{DATADIR}/ptSource-ind3-14-195keV/model_files'
MAPDIR = f'{DIR}/hd5_files'
FITSDIR = f'{DIR}/fits_files'
CSV='${CSV}'

df = pd.read_csv(CSV,sep='\\s+').to_numpy()
sourceName = df[:,0]
RA = df[:,1]
Dec = df[:,2]

for i,source in enumerate(sourceName):
    filename_hd5 = f'{MAPDIR}/{source}_model_map.hd5'
    print(f"Filename: {filename_hd5} for {source}")
    
    output_fits = f'{FITSDIR}/{source}'
    print(f"Output header: {output_fits}")    

    ret = os.system(
        f"python hal_hdf5_to_fits.py --input {filename_hd5} --output {output_fits}"
    )
    
    if ret != 0:
        print(f"WARNING: hal_hdf5_to_fits.py failed for {source}", file=sys.stderr)

print("Done! All FITS files created")

EOF

#mamba deactivate 
#source /data/disk01/home/zylaphoe/hawc_software/init_aerie.sh 
#
#
### AERIE -- combine maps and plot
#python <<EOF 
#import os, sys
#import pandas as pd
#
#DATADIR = '${DATADIR}'
#DIR = f'{DATADIR}/ptSource-ind3-14-195keV'
#MAPDIR = f'{DIR}/map_files'
#FITSDIR = f'{DIR}/fits_files'
#PNGDIR = f'{DIR}/png_files'
#CSV = '${CSV}'
#
#
#df = pd.read_csv(CSV, sep=r'\s+').to_numpy()
#sourceName = df[35:, 0]
#RA = df[:, 1]
#Dec = df[:, 2]
#
#data_radius = 5.
#
#bins = [
#    'B2C0','B2C1','B3C0','B3C1','B4C0','B4C1',
#    'B5C0','B5C1','B6C0','B6C1','B7C0','B7C1',
#    'B8C0','B8C1','B9C0','B9C1','B10C0','B10C1'
#]  
#
#for i, source in enumerate(sourceName):
#    filelist = [
#        f"{FITSDIR}/{source}_bin{binNo}.fits.gz"
#        for binNo in bins
#    ]
#    print(f'List of files: {filelist}')
#    
#    combo = f'{FITSDIR}/{source}_comb'
#    print(f'Combined file: {combo}')
#
#    ret = os.system(
#        f"aerie-apps-combine-maps --inputs {' '.join(filelist)} -o {combined}"
#    )
#    if ret != 0:
#        print(f"WARNING: combine-maps failed for {source}", file=sys.stderr)
#
#
#print("All combined FITS files created")
#
#for i, source in enumerate(sourceName):
#    ra = RA[i]
#    dec = Dec[i]
#
#    combo = f'{FITSDIR}/{source}_comb.fits.gz'
#    print(f"Combined file: {combo}")
#    
#    output_png = f'{PNGDIR}/{source}.png'
#    print(f'Output file: {output_png}')
#
#    ret = os.system(
#        f"plotMercator.py {combo} --origin {ra} {dec} {data_radius} {data_radius} -o {output_png}"
#    )
#    if ret!= 0: 
#        print(f"WARNING: plotMercatory.py failed for {source}", file=sys.stderr)
#print("All PNGs created")
#
#EOF
