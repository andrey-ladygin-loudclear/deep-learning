
df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
df = df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average})
pd.cut(df['avg'],10)


# # Pivot Tables

# In[ ]:

#http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
df = pd.read_csv('cars.csv')


# In[ ]:

df.head()


# In[ ]:

df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)


# In[ ]:

df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean,np.min], margins=True)
