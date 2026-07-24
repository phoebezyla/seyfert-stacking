#!/bin/sh
#SBATCH --time=30:00:00 --mem-per-cpu=8000mb
#
# RUN:
# __RUN__

source /data/disk01/home/zylaphoe/micromamba/etc/profile.d/mamba.sh
mamba activate new_hal

#python make-model-hal.py > model_out.txt
#python ind-27llh-profiles.py > out-27ind.txt 
python adding_27ixsources.py > out-27add.txt
##python stack-seyferts.py > sy_out.txt
