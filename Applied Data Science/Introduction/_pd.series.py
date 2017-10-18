import pandas as pd
animals = ['Tiger', 'Bear', 'Mouse']
print(pd.Series(animals))
print(pd.Series([1, 2, 3]))


import numpy as np
np.nan == np.nan# false
np.isnan(np.nan)



sports =  {
    'sea': 'ship 1',
    'golf': 'bal',
    'car': 'infinity'
}

s = pd.Series(sports)
print(s)
print(s.index)

s = pd.Series(['tiger', 'bear', 'moose'], index=['india', 'america', 'canada'])