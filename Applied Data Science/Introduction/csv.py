import csv
with open('mpg.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))


print(mpg[:3])
print(len(mpg))
print(mpg[0].keys())

s = sum(float(d['city']) for d in mpg) / len(mpg)

cylinders = set(d['cyl'] for d in mpg)