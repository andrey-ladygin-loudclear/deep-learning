import pandas as pd

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevin',
                        'Item Purchased': 'Kitty Letter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1,purchase_2,purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
print(df.head())
print('-----------------------------------')
#data from store 2
print(df.loc['Store 1'])
print('-----------------------------------')
print(df.loc['Store 2', 'Cost'])
print('-----------------------------------')
print(df.T)
print('-----------------------------------')

cp_df = df.copy()
df.drop('Store 1') # return a copy with removed rows
del cp_df['Name'] # delete data at this variable

#add new data
df['Location'] = None

costs = df['Cost']
print(costs)
costs += 2 # will change prices in dataframe
print(df)