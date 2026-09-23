import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv
from stacking_functions import *

df_S = pd.read_csv("data_normalized.csv",sep=',').to_numpy()
sourceName = df_S[:,0]

IX   = ['2','27','3']
inds = [2.0, 2.7,3.0]


dfs = {ix: pd.read_csv(f"results_csvs/normed-long-stacked-ind{ix}.csv") for ix in IX}

PhoebePlotting.SpectrumPlots(dfs,"stacked51_long_spectrum")
PhoebePlotting.LikelihoodPlots(dfs,"stacked51_long_lh",xlims=(1e-30,1e-15))

keys = list(dfs.keys())
print(keys)

#begin_TS = np.array([-0.020819783210754395,-0.022848963737487793,-0.02813541889190674])
begin_TS = np.array([0,0,0])

c = plt.cm.tab10(np.linspace(0,1,len(keys)))
plt.figure(layout='constrained')#,figsize=(8,20))

for i, key in enumerate(keys):
    df = dfs[key]
    pivots = df['pivot'].values.astype(float)
    ts     = df['TS'].values.astype(float) - begin_TS[i]

    plt.scatter(pivots,ts,color=c[i],label=f"Ind = {inds[i]}")

plt.xlabel("Energy [TeV]")
plt.ylabel("TS")
plt.yscale('log')
plt.legend()
plt.grid()
plt.title(f"Comparison of stacked TS values for 50 sources")
plt.savefig(f"normed_long_ts_comp_51.png")
plt.close()
