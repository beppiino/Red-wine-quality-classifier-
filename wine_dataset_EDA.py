# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot(x_value, y_value):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    sns.barplot(x=x_value, y=y_value, data=wine_data, ax=ax[0], palette='RdBu_r')
    sns.boxplot(x= x_value, y= y_value, data=wine_data, ax=ax[1],  palette='RdBu_r')

#segment data in two bins 
#bad ==> 1, 2, 3 ,4 , 5, 6
#good ==> 7, 8, 9, 10
def manipulate_data(): 
    bins = (2, 6.5, 8)
    group_names = ['bad', 'good']
    wine_data['quality'] = pd.cut(wine_data['quality'], bins = bins, labels = group_names)
    return wine_data

wine_data = pd.read_csv("C:/Users/user/Desktop/input/winequality-red.csv")

#dataset visualitazion 
print('Number of rows in the dataset: ', wine_data.shape[0])
print('Number of columns in the dataset: ', wine_data.shape[1])
print(wine_data.isnull().sum())
wine_data.describe()
wine_data['quality'].unique()
sns.countplot(wine_data['quality'], palette='RdBu_r')

plot('quality', 'fixed acidity')
plot('quality', 'volatile acidity')
plot('quality', 'citric acid')
plot('quality', 'residual sugar')
plot('quality', 'chlorides')
plot('quality', 'free sulfur dioxide')
plot('quality', 'sulphates')
plot('quality', 'alcohol')


correlations = wine_data.corr()['quality'].drop('quality')
print(correlations)
plt.figure(figsize=(12,8))
corr = wine_data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,mask=mask, annot=True, linewidths=1, cmap='RdBu_r')
plt.show()

#preprocessing
manipulate_data()
plt.figure(figsize=(7,6))
sns.countplot(x='quality', data=wine_data, palette='RdBu_r')
plt.show()


