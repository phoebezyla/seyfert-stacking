from astropy import units as u
import matplotlib.pyplot as plt
import os, sys, time
import csv
import warnings
import yaml
import numpy as np
import pandas as pd


## Set paths ##
DATADIR = '/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/'
DIR = '/lustre/hawcz01/scratch/userspace/zylaphoe/seyfert/ptSource-ind3-14-195keV/'

## Load CSV and initialize arrays ##
df = pd.read_csv("data_normalized.csv",sep=',').to_numpy()

sourceName = df[:,0]
RA    = df[:,1]
Dec   = df[:,2]
A     = df[:,3]
Anorm = df[:,4]

plt.scatter(Dec,Anorm,c='r',s=20)
plt.yscale('log')
plt.ylabel(r"Source Weight (Normalized X-Ray Flux")
plt.xlabel("Declination [deg]")
plt.title("Normalized Weighing factor vs Declination")
plt.grid()
for i, txt in enumerate(sourceName):
    plt.annotate(txt, (Dec[i], A[i]), textcoords="offset points", xytext=(0,10),ha='center')
plt.savefig("weights_normed_dec_log_annotate.png")
plt.close()


