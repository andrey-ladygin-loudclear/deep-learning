x=(1, 'a', 2, 'b') #can not be changed
print(type(x))

x=[1, 'a', 2, 'b']
x.append(3.3)
print(type(x))

[1, 2] + [4, 5]
[1] * 3 # [1,1,1]


x = {'1': 2, 'Bill': 'emais'}
x['1']
x['1'] = '123123'

for name in x: print(x[name])
for val in x.values(): print(val)
for name, val in x.items(): print(name, val)


sales = {
    'price': 3.24,
    'num_items': 4,
    'person': 'Chris'
}

statement = '{} bought {} item(s) at a price of {} each for a total of {}'

print(statement.format(sales['pearson'],
                       sales['num_items'],
                       sales['price'],
                       sales['num_items']*sales['price']))