import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# read the csv file
name='A2-personalized/raisin'

df= pd.read_csv(name+'.csv',sep=';',header=1)
print(df.head())
print(df.describe())

df_processed = pd.DataFrame()

for column in df.columns:
    df_processed[column] = df[column].replace('unknown', df[column].mode()[0])
    is_categorical = df_processed[column].isin(['Kecimen', 'Besni']).all()
    if is_categorical:
        df_processed[column] = df_processed[column].replace({'Kecimen': 1, 'Besni': 0}).astype(int)
    else:
        is_numeric = pd.to_numeric(df_processed[column], errors='coerce').notnull().all()
        if not is_numeric:
            unique=df_processed[column].unique()
            unique={k:v for v,k in enumerate(unique)}
            df_processed[column]=df_processed[column].map(unique)

#Normalization
columns = df_processed.columns
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_processed[columns]), columns=columns)

df_normalized = df_normalized.sample(frac=1, random_state=42)

print(df_normalized.head())

output_file_name=name+'-processed.csv'
df_normalized.to_csv(output_file_name,sep=',', index=False,header=None)
