import pandas as pd
import numpy as np

pd.Timestamp('9/1/2016 10:05AM')

pd.Period('1/2016')

pd.Period('3/5/2016')

t1 = pd.Series(list('abc'), [pd.Timestamp('2016-01-01'),pd.Timestamp('2016-01-02'),pd.Timestamp('2016-01-03')])
print(t1)
print(t1.index)

t2 = pd.Series(list('abc'), [pd.Period('2016-01'),pd.Period('2016-01'), pd.Period('2016-01')])
print(t2)
print(t2.index)

d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))

ts3.index = pd.to_datetime(ts3.index)


dates = pd.date_range('10-01-2016', periods=9, freq='2W-SUN')