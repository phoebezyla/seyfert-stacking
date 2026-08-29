import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv

df_S = pd.read_csv("data50sources.csv",sep='\\s+').to_numpy()

sourceName = df_S[:,0]

df = {}
begin_TS = np.array([-0.020819783210754395,-0.022848963737487793,-0.02813541889190674])

pivot = np.array([1.0,5.0,10.0])  # TeV
#ind = '3'

plt.figure(layout='constrained',figsize=(8,20))
ix = ['2','27','3']
c = cm.prism(np.linspace(0,1,len(sourceName)))

#for name in sourceName:
for ind in ix:
    ts_values = []
    pivots    = []

    with open("mod-nopiv-results-stacked-ind%s-50sources.csv"%(ind),newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
#            if row['excludedSource'] == name:
            pivots.append(float(row['pivot']))
            ts_values.append(float(row['TS']))
    ts_values = np.array(ts_values)
    df[ind] = {
    #df[name] = {
          'pivot': pivots,
          "ts": ts_values-begin_TS,
    }
    
#for i, name in enumerate(sourceName):
for i, ind in enumerate(ix):
    plt.scatter(pivot,df[ind]['ts'],color=c[i],label=f"Ind = {ind}")#"Excluding {name}")
    #for x,y in zip(df[name]['pivot'],df[name]['ts']):
    #    plt.text(x,y,f"- {name}",fontsize=6,color=c[i],ha='left',va='bottom')


plt.xlabel("Energy [TeV]")
plt.ylabel("TS")
plt.yscale('log')
plt.legend()
plt.grid()
plt.title(f"Comparison of stacked TS values for 50 sources")#, ind = {ind} (excluding one source per)")
plt.savefig(f"mod_nopiv_ts_comp_50.png")
plt.close()
