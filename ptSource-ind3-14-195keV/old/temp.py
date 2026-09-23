import numpy as np

midE = np.array([1.0,5.0,10.0])  # TeV
lowerE = midE/np.power(10,0.25)
upperE = np.power(10,np.log10(lowerE+0.5))

print(midE[0],lowerE[0],upperE[0])
print(midE[1],lowerE[1],upperE[1])
print(midE[2],lowerE[2],upperE[2])
