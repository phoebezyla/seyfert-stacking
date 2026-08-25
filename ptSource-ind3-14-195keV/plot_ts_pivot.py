import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv

dfSources = pd.read_csv("dataSus.csv",sep='\\s+').to_numpy()

sourceName = dfSources[:,0]
RA = dfSources[:,1]
Dec = dfSources[:,2]
A = dfSources[:,3]
names = dfSources[:,4]

df = {}
begin_TS = []

pivot = np.array([1.0,5.0,10.0])  # TeV
c = cm.tab10(range(len(sourceName)))

with open("results-stacked-ind3-mod.csv",newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        begin_TS.append(float(row['TS']))

for i, name in enumerate(names):
    ts_values = []
    with open("results-stacked-ind3-%s.csv"%(name),newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_values.append(float(row['TS']))
        df[sourceName[i]] = {"ts": ts_values}

plt.figure(layout='constrained')

for i, name in enumerate(sourceName):
    plt.scatter(pivot,df[name]['ts'],color=c[i],label=f"{name}")

plt.xlabel("Energy [TeV]")
plt.ylabel("TS")
plt.legend()
plt.grid()
plt.title("Comparison of Flux values for suspect sources")
plt.savefig("ts_comp_suspect_nine.png")
plt.close()
