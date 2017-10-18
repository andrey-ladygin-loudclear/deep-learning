import pandas as pd
import numpy as np
import time

sports = {
    'Archery': 'Bhutan',
    'golf': 'scotland',
    'car': 'europe',
    'kill': 'indonesia'
}

s = pd.Series(sports)
print(s.iloc[3])
print(s.loc['golf'])
print(s[3])
print(s['golf'])


s = pd.Series([100.00, 120.00, 101.00, 3.00])
total = 0
for item in s:
    total += item
print(total)
print(np.sum(s))
print('random:')

s = pd.Series(np.random.randint(0, 1000, 100000))
print(s.head()) # return first 5 elements

start_time = time.time()
summary = 0
for it in s: summary += it
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
summary = np.sum(summary)
print("--- %s seconds ---" % (time.time() - start_time))


s += 2 # would be easy ranther than add values by iterations