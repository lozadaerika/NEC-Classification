# read th information of a CSV and load into a dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the csv file
name='A2-bank/bank-additional'

df= pd.read_csv(name+'.csv',sep=';')
print(df.head())

df_processed = pd.DataFrame()

# Replace 'unknown' with the mode in each column
for column in df.columns:
    mode_value = df[column].mode()[0]
    df_processed[column] = df[column].replace('unknown', mode_value)
    is_yes_no = df_processed[column].isin(['yes', 'no']).all()
    if is_yes_no:
        df_processed[column] = df_processed[column].replace({'yes': 1, 'no': 0})
    else:
        is_numeric = pd.to_numeric(df_processed[column], errors='coerce').notnull().all()
        if not is_numeric:
            #obtain all the values of the column provincia and create a dictionary
            unique=df_processed[column].unique()
            #transform the list to a dictionary
            unique={k:v for v,k in enumerate(unique)}
            print(unique)
            df_processed[column]=df_processed[column].map(unique)
            #df_all2.head()    

        print(column, is_numeric,is_yes_no)

output_file_name=name+'-processed.csv'
df_processed.to_csv(output_file_name, index=False)
""" 
#obtain all the values of the column provincia and create a dictionary
provincias=df_all['Provincia'].unique()
#transform the list to a dictionary
provincias={k:v for v,k in enumerate(provincias)}
print(provincias)

df_all2['Provincia2']=df_all2['Provincia'].map(provincias)

df_all2.head()
 """

#print("\nDataFrame has been saved to " + output_file_name)

#print(df.isnull().sum())
#replace the missing values with the mean of the column for tempratura media
#df_all2['Temperatura media (ºC)'].fillna(df_all2['Temperatura media (ºC)'].mode(),inplace=True)
#print(df_all2.isnull().sum())