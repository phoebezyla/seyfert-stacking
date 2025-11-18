#!/bin/bash
#
#

source /data/disk01/home/zylaphoe/micromamba/etc/profile.d/mamba.sh
mamba activate new_hal


python stack-seyferts.py > sy_out.txt
