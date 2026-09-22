import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

IX = ['2','27','3']
inds = [2.0,2.7,3.0]
NUM = ['one','two']
c = plt.cm.tab10(np.linspace(0, 1, len(NUM)*len(IX))).reshape(len(NUM), len(IX), 4) 

xarr = np.linspace(0.5,15,10000)  # in TeV


plt.figure(layout='constrained')

for j,ix in enumerate(IX):
    for i,num in enumerate(NUM):
        df = pd.read_csv(f"mod-nopiv-stacked-ind{ix}-{num}crab-results.csv")
        pivot = df['pivot']
        indminNorm = df['indminNorm']

        plt.scatter(pivot,indminNorm,color=c[i,j],label=f"Index = {ix}, {num} crabs")
   
        for k,pv in enumerate(pivot):
            y = indminNorm[k] * (xarr/pv) ** -(inds[j]) 
            plt.plot(xarr,y,color=c[i,j],label=f"Powerlaw Spectrum Line")


plt.xlabel("Pivot energy [TeV]")
plt.ylabel("Normalization")
plt.title("Crab calculated spectrum")
plt.legend()
plt.savefig("crab_spec_calc.png")
plt.close()

