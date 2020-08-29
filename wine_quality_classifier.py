# -*- coding: utf-8 -*-
"""

@author: Daniela Grassi
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


wine_data = pd.read_csv("C:/Users/user/Desktop/input/winequality-red.csv")
group_names = ['bad', 'good']

def plot_results(classifier,X_test,y_test, name):
    
    pred = classifier.predict(X_test)
    print(classification_report(y_test, pred)) 
    cm = confusion_matrix(y_test, pred, group_names)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap="Reds"); 

    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(name); 
    ax.xaxis.set_ticklabels(group_names); ax.yaxis.set_ticklabels(group_names);
    plt.show()

#plot the boxplot for visualization of correlation 
def plot(x_value, y_value):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(x= x_value, y= y_value, data=wine_data,  palette='RdBu_r')
    

#segment data in two bins 
#bad ==> 1, 2, 3 ,4 , 5, 
#good ==>6, 7, 8, 9, 10
def manipulate_data(): 
    bins = (2, 5.5, 8)
    wine_data['quality'] = pd.cut(wine_data['quality'], bins = bins, labels = group_names)
    return wine_data

#import the dataset
wine_data = pd.read_csv("C:/Users/user/Desktop/input/winequality-red.csv")
print(wine_data['quality'].value_counts())

# %%% dataset visualitazion %%%% 
print('Number of rows in the dataset: ', wine_data.shape[0])
print('Number of columns in the dataset: ', wine_data.shape[1])
print(wine_data.isnull().sum())
wine_data.describe()
wine_data['quality'].unique()
sns.countplot(wine_data['quality'], palette='RdBu_r')

#data correlation
plot('quality', 'fixed acidity')
plot('quality', 'volatile acidity')
plot('quality', 'citric acid')
plot('quality', 'residual sugar')
plot('quality', 'chlorides')
plot('quality', 'free sulfur dioxide')
plot('quality', 'sulphates')
plot('quality', 'alcohol')
plot('quality', 'pH')
plot('quality', 'density')
plot('quality', 'total sulfur dioxide')
correlations = wine_data.corr()['quality'].drop('quality')
print(correlations)
plt.figure(figsize=(12,8))
corr = wine_data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,mask=mask, annot=True, linewidths=1, cmap='RdBu_r')
plt.show()

# %%% preprocessing %%%
manipulate_data()
sns.countplot(x='quality', data=wine_data, palette='RdBu_r')
plt.show()

#subset x with all feature without 'quality' and subset y with feature quality
X = wine_data.iloc[:,:11]
y = wine_data['quality']

#normalize
sc = StandardScaler()
X = sc.fit_transform(X)

#principal component analysis
from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(X)
variance = pca.explained_variance_ratio_ #calculate variance ratios
var=np.cumsum(np.round(variance, decimals=3)*100)
print(var)

plt.figure(figsize=(7,6))
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.plot(var, 'ro-')
plt.grid()

#As per the graph,there are 8 principal components for 90% of variation in the data. 
#pick the first 8 components for prediction.
pca_new = PCA(n_components=8)
X_new = pca_new.fit_transform(X)
print(wine_data['quality'].value_counts())

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=.25, random_state=0)
print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)



# %%% Classification %%%%

# Support Vector Machines
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf', 'sigmoid']}

grid_svm = GridSearchCV(SVC(), param_grid=param_grid, cv=5, refit = True, verbose = False) 
grid_svm.fit(X_train, y_train) 
print("best_params", grid_svm.best_params_) 
print("best_estimator", grid_svm.best_estimator_) 
plot_results(grid_svm, X_test, y_test,'Support Vector Machines')


#Random Forest
rf=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [100, 250, 500],
    'max_features': ['auto', 'log2'],
    'criterion' :['gini', 'entropy']
}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5, refit = True, verbose = False)
grid_rf.fit(X_train, y_train)

print("best_params", grid_rf.best_params_) 
print("best_estimator", grid_rf.best_estimator_) 
plot_results(grid_rf, X_test, y_test,'Random Forest')

#K-nearest neighbors
knn = KNeighborsClassifier()
param_grid = {'n_neighbors':[1,4,5,6,7,8],
              'leaf_size':[1,3,5,10],
}

grid_knn = GridSearchCV(knn, param_grid=param_grid)
grid_knn.fit(X_train ,y_train)

print("best_params", grid_knn.best_params_) 
print("best_estimator", grid_knn.best_estimator_) 
plot_results(grid_knn, X_test, y_test,'K-nearest neighbors')

# AdaBoost
Ada = AdaBoostClassifier(random_state=1)
Ada.fit(X_train, y_train)
plot_results(Ada, X_test, y_test,'AdaBoost')


#Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
plot_results(gaussian, X_test, y_test,'Gaussian Naive Bayes')

