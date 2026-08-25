import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv

df = {}
begin_TS = np.array([-0.020819783210754395,-0.022848963737487793,-0.02813541889190674])

pivot = np.array([1.0,5.0,10.0])  # TeV

plt.figure(layout='constrained')
ix = ['2','27','3']
c = cm.tab10(range(len(ix)))

for ind in ix:
    ts_values = []
    with open("results-stacked-ind%s-50sources.csv"%(ind),newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_values.append(float(row['TS']))
        df[ind] = {
          "ts": ts_values-begin_TS,
        }
    
for i, name in enumerate(ix):
    plt.scatter(pivot,df[name]['ts'],color=c[i],label=f"Index = {name}")

plt.xlabel("Energy [TeV]")
plt.ylabel("TS")
plt.yscale('log')
plt.legend()
plt.grid()
plt.title("Comparison of stacked TS values for 50 sources")
plt.savefig("ts_comp_no_ngc4151_logy.png")
plt.close()
