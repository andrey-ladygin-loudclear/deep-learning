import pandas as pd
import numpy as np


df = pd.DataFrame([{'Name': 'Chris', 'role': 'Director'},
                {'Name': 'Kevin 4', 'role': 'Movie'},
                {'Name': 'Vinod 5', 'role': 'Animator'}])
df2 = pd.DataFrame([{'Name': 'Chris', 'University': '9a'},
                {'Name': 'Kevin 2', 'University': '10b'},
                {'Name': 'Vinod 3', 'University': '12a'}])

df = df.set_index('Name')
df2 = df2.set_index('Name')

new = pd.merge(df, df2, how='outer', left_index=True, right_index=True)
print(new)
new = pd.merge(df, df2, how='inner', left_index=True, right_index=True)
print(new)

pd.merge(df, df2, how='inner', left_on=['First name', 'Last Name'], right_on=['First name', 'Last Name'])




def min_max(row):
    data = row[['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})

df.apply(min_max, axis=1) # start from 1

def min_max(row):
    data = row[['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']]
    row['max'] = np.max(data)
    row['min'] = np.min(data)
    return row

df.apply(min_max, axis=1) # start from 1

rows = ['POPESTIMATE2010',
     'POPESTIMATE2011',
     'POPESTIMATE2012',
     'POPESTIMATE2013',
     'POPESTIMATE2014',
     'POPESTIMATE2015']

df.apply(lambda x: np.max(x[rows]), axis=1)