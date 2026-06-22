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
df = pd.read_csv("data14-195.csv",sep='\\s+').to_numpy()

sourceName = df[:,0]
RA = df[:,1]
Dec = df[:,2]
A = df[:,3]

results_df = {}
TSarr = []

with open("ind-results-allscs.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        source_name = row["sourceName"]
        results_df[source_name] = {
            "nullLLH": float(row["nullLLH"]),
            "alt_hyp": float(row["alt_hyp"]),
            "TS":      float(row["TS"]),
            "log_val": [float(row[f"log_val_{k}"]) for k in range(200)],
            "norms":   [float(row[f"norms_{k}"]) for k in range(200)],
        }

for k,v in results_df.items():
    TSarr.append(v["TS"])

#plt.scatter(A,TSarr,c='r',s=20)
#plt.xlabel(r"X-Ray Flux (14-195 keV) [$10^{-12}$ ergs s$^{-1}$ cm$^{-2}$]")
#plt.ylabel("TS")
#plt.title("TS versus Weighing factor")
#plt.grid()
#plt.savefig("ts_weights_scatter.png")
#plt.close()

plt.scatter(A,TSarr,c='r',s=20)
plt.yscale('log')
plt.xlabel(r"X-Ray Flux (14-195 keV) [$10^{-12}$ ergs s$^{-1}$ cm$^{-2}$]")
plt.ylabel("TS")
plt.title("TS versus Weighing factor")
plt.grid()
for i, txt in enumerate(sourceName):
    plt.annotate(txt,(A[i], TSarr[i]), textcoords="offset points",
        xytext=(0, 10), ha='center')

plt.savefig("ts_weights_annotate.png")
plt.close()


