import pandas as pd

frame = pd.read_csv('data/dataset.tsv', sep='\t', header=0)
print frame.dtypes
frame.Birth = frame.Birth.apply(pd.to_datetime)
print frame.dtypes
print frame.info()
frame.fillna('diffworker', inplace=True)
print frame.Position
print frame[['Name', 'Position']]
print frame.head(3)
print frame[:3] #first 3 records
print frame[-3:] #last 3 records
print frame.loc[[1,3,5], ['Name', 'City']]
print frame.iloc[[1,3,5], [0,2]]
print frame.ix[[1,3,5], ['Name', 'City']]
print frame.ix[[1,3,5], [0,2]]

print frame[frame.Birth >= pd.datetime(1985,1,1)]
print frame[(frame.Birth >= pd.datetime(1985,1,1)) & (frame.City != 'Moscov')]
print frame[(frame.Birth >= pd.datetime(1985,1,1)) | (frame.City != 'Moscov')]