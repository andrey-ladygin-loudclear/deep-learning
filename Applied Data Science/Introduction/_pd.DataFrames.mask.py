import pandas as pd

df = pd.read_csv('worldcitiespop.txt')
only_gold = df.where(df['Gold'] > 0)
only_gold['Gold'].count()
only_gold = only_gold.dropna()


gold_len = len(df[(df['Gold']>0) | df['Gold.1']>0])
print(gold_len)

res = df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]
print(res)