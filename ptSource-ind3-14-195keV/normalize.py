import pandas as pd

df = pd.read_csv("data14-195.csv",sep='\\s+')

A = df.iloc[:,3]
Anorm = []

df['Normed_weights'] = (A-min(A)) / (max(A)-min(A))

print(df['Normed_weights'])

df.to_csv('data_normalized.csv',index=False)
