import pandas as pd

df = pd.read_csv('worldcitiespop.txt')
print(df.head())
df = pd.read_csv('worldcitiespop.txt', index_col=0, skiprows=1)
print(df.head())


for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True)

df.head()



df['country'] = df.index
df = df.set_index('Gold')


df['SOMEVAL'].unique()

df = df[df['SOMEVAL'] == 50]

df.loc[ [('Colum 1', 'Colum 2'), ('Colum 1', 'Colum 3')] ]



########### INDEX ##########
df = df.set_index(['time', 'user'])
df = df.sort_index()
df