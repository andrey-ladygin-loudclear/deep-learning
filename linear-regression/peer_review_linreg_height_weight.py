#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('weights_heights.csv', index_col='Index')
#data.plot(y='Height', kind='hist', color='red',  title='Height (inch.) distribution')
#plt.show()

'''
Один из эффективных методов первичного анализа данных - отображение попарных зависимостей признаков.
Создается  m×mm×m  графиков (m - число признаков), где по диагонали рисуются гистограммы распределения признаков,
а вне диагонали - scatter plots зависимости двух признаков. Это можно делать с помощью метода  scatter_matrixscatter_matrix
Pandas Data Frame или pairplot библиотеки Seaborn.
Чтобы проиллюстрировать этот метод, интересней добавить третий признак. Создадим признак Индекс массы тела (BMI).
Для этого воспользуемся удобной связкой метода apply Pandas DataFrame и lambda-функций Python.
'''

def make_bmi(height_inch, weight_pound):
    METER_TO_INCH, KILO_TO_POUND = 39.37, 2.20462
    return (weight_pound / KILO_TO_POUND) / (height_inch / METER_TO_INCH) ** 2

data['BMI'] = data.apply(lambda row: make_bmi(row['Height'], row['Weight']), axis=1)

#sns.pairplot(data)
#plt.show()


'''
Часто при первичном анализе данных надо исследовать зависимость какого-то количественного признака от категориального
(скажем, зарплаты от пола сотрудника). В этом помогут "ящики с усами" - boxplots библиотеки Seaborn.
Box plot - это компактный способ показать статистики вещественного признака (среднее и квартили)
по разным значениям категориального признака. Также помогает отслеживать "выбросы" - наблюдения,
в которых значение данного вещественного признака сильно отличается от других.
[4]. Создайте в DataFrame data новый признак weight_category, который будет иметь 3 значения:
1 – если вес меньше 120 фунтов. (~ 54 кг.), 3 - если вес больше или равен 150 фунтов (~68 кг.),
2 – в остальных случаях. Постройте «ящик с усами» (boxplot), демонстрирующий зависимость роста от весовой категории.
Используйте метод boxplot библиотеки Seaborn и метод apply Pandas DataFrame.
Подпишите ось y меткой «Рост», ось x – меткой «Весовая категория».
'''

def weight_category(weight):
    return 1 if weight < 120 else 3 if weight >= 150 else 2

data['weight_cat'] = data['Weight'].apply(weight_category)

'''
Постройте scatter plot зависимости роста от веса, используя метод plot для Pandas DataFrame с аргументом kind='scatter'. Подпишите картинку.
'''

data.plot(y='Height', x='weight_cat', kind='scatter', color='red',  title='Height (inch.) distribution')
plt.show()