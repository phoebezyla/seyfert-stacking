import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv

dfSources = pd.read_csv("data14-195.csv",sep='\\s+').to_numpy()

sourceName = dfSources[:,0]
RA = dfSources[:,1]
Dec = dfSources[:,2]
A = dfSources[:,3]

df = {}
begin_TS = np.array([-0.020819783210754395,-0.022848963737487793,-0.02813541889190674])

pivot = np.array([1.0,5.0,10.0])  # TeV
c = cm.prism(np.linspace(0, 1, len(sourceName)))


for i, name in enumerate(sourceName):
    ts_values = []
    with open("results-stacked-ind3-%s.csv"%(name),newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_values.append(float(row['TS']))
        df[sourceName[i]] = {"ts": ts_values-begin_TS}

plt.figure(layout='constrained',figsize=(8,10))

for i, name in enumerate(sourceName):
    plt.scatter(pivot,df[name]['ts'],color=c[i],label=f"{name}")
    for x, y in zip(pivot, df[name]['ts']):
        plt.annotate(name,
                     xy=(x, y),
                     xytext=(5, 0), textcoords='offset points',
                     fontsize=8, color=c[i], va='center')

plt.xlabel("Energy [TeV]")
plt.ylabel("TS")
plt.yscale('log')
#plt.legend()
plt.grid()
plt.title("Comparison of TS values for suspect sources")
plt.savefig("ts_comp.png")
plt.close()
