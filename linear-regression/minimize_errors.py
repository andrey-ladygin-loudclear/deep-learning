#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('weights_heights.csv', index_col='Index')

'''
В простейшей постановке задача прогноза значения вещественного признака по прочим признакам
(задача восстановления регрессии) решается минимизацией квадратичной функции ошибки.
[6]. Напишите функцию, которая по двум параметрам  w0w0  и  w1w1  вычисляет квадратичную ошибку
приближения зависимости роста  yy  от веса  xx прямой линией  y=w0+w1∗xy=w0+w1∗x:
'''

def error(w0, w1):
    sum = 0
    for index, row in data.iterrows():
        sum += (row['Height'] - (w0 + w1 * row['Weight'])) ** 2

    return sum

'''
Итак, мы решаем задачу: как через облако точек, соответсвующих наблюдениям в нашем наборе данных,
в пространстве признаков "Рост" и "Вес" провести прямую линию так, чтобы минимизировать функционал из п. 6.
Для начала давайте отобразим хоть какие-то прямые и убедимся, что они плохо передают зависимость роста от веса.
[7]. Проведите на графике из п. 5 Задания 1 две прямые, соответствующие значениям параметров
( w0,w1)=(60,0.05)w0,w1)=(60,0.05)  и ( w0,w1)=(50,0.16)w0,w1)=(50,0.16) . Используйте метод plot из matplotlib.pyplot,
а также метод linspace библиотеки NumPy. Подпишите оси и график.
'''

def line(w0, w1, x):
    return w0 + w1 * x

w01 = 60
w11 = 0.05
w02 = 50
w12 = 0.16


line1 = [line(w01, w11, i) for i in range(5)]
line2 = [line(w02, w12, i) for i in range(5)]

#error(1,2)

def weight_category(weight):
    return 1 if weight < 120 else 3 if weight >= 150 else 2

# data['weight_cat'] = data['Weight'].apply(weight_category)
# data.plot(y='Height', x='weight_cat', kind='scatter', color='red',  title='Height (inch.) distribution')
#
# plt.plot(line1, color='b', linestyle='-', linewidth=2)
# plt.plot(line2, color='g', linestyle='-', linewidth=2)
# plt.show()


w0 = 50
err = [error(w0, i) for i in range(20)]
print err
plt.plot(err, color='g', linestyle='-', linewidth=2)
plt.show()