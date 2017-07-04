import pandas as pd

frame = pd.DataFrame({'numbers': range(10), 'chars': ['a']*10})
print frame
#frame = pd.read_csv('dataset.tsv', header=0, sep='\t')
print frame.columns
print frame.shape
new_line = {'Name': 'Perov', 'Birth': '22.03.1990', 'City':'Penza'}
frame = frame.append(new_line, ignore_index=True)
frame['IsStudent'] = [False] * 5 + [True] * 2
#frame = frame.drop([5,6], axis=0)
frame.drop([5,6], axis=0, inplace=True)
frame.drop('IsStudent', axis=1, inplace=True)
frame.to_csv('updated_dataset.csv', sep=',', header=True, index=None)