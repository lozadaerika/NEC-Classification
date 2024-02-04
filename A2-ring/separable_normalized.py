import pandas as pd

name='A2-ring/A2-ring-separable'

df= pd.read_csv(name+'.txt',sep='\t',header=None)

print(df.describe())
print(df.head())

df_processed=df.copy()

# Standardization
df_processed.iloc[:, 0]= (df.iloc[:,0] - df.iloc[:,0].mean()) / df.iloc[:,0].std()
df_processed.iloc[:, 1]= (df.iloc[:,1] - df.iloc[:,1].mean()) / df.iloc[:,1].std()

output_file_name=name+'-normalized.csv'
df_processed.to_csv(output_file_name,sep=',', index=False,header=None)
