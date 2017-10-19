import pandas as pd
import numpy as np


df = pd.DataFrame([{'Name': 'Chris', 'role': 'Director'},
                   {'Name': 'Kevin 4', 'role': 'Movie'},
                   {'Name': 'Vinod 5', 'role': 'Animator'}])

for group, frame in df.groupby('SOMENAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('Counties in state %s have an average of %s' % (group, avg))



df = df.set_index('STNAME')

def fun(item):
    if item[0] < 'M':
        return 0
    if item[0] < 'Q':
        return 1
    return 2

for group, frame in df.groupby(fun):
    print(len(frame), group)
    #print('Counties in state %s have an average of %s' % (group, avg))


df.groupby('STNAME').agg({'CENSUSS2010POP': np.average})
df.set_index('STNAME').groupby(level=0)['CENANSUS2010POP'].agg({'avg': np.average, 'sum': np.sum})




df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]


# In[ ]:

df.groupby('STNAME').agg({'CENSUS2010POP': np.average})


# In[ ]:

print(type(df.groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']))
print(type(df.groupby(level=0)['POPESTIMATE2010']))


# In[ ]:

(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP']
 .agg({'avg': np.average, 'sum': np.sum}))


# In[ ]:

(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
 .agg({'avg': np.average, 'sum': np.sum}))


# In[ ]:

(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
 .agg({'POPESTIMATE2010': np.average, 'POPESTIMATE2011': np.sum}))