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
#import the dataset


wine_data = pd.read_csv("C:/Users/user/Desktop/input/winequality-red.csv")
print(wine_data['quality'].value_counts())
group_names = ['bad', 'good']

def plot_results(classifier,X_test,y_test, name):
    
    pred = classifier.predict(X_test)
    print(classification_report(y_test, pred)) 
    cm = confusion_matrix(y_test, pred, group_names)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap="Reds"); #annot=True to annotate cells

    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(name); 
    ax.xaxis.set_ticklabels(group_names); ax.yaxis.set_ticklabels(group_names);
    plt.show()




#preprocessing
bins = (2, 6.5, 8)
wine_data['quality'] = pd.cut(wine_data['quality'], bins = bins, labels = group_names)


#subset x with all feature without 'quality' and subset y with feature quality
X = wine_data.iloc[:,:11]
y = wine_data['quality']

#normalize
sc = StandardScaler()
X = sc.fit_transform(X)



# esegue sovracampionamento usando SMOTE


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

#As per the graph, we can see that 8 principal components attribute for 90% of variation in the data. 
#we shall pick the first 8 components for our prediction.
pca_new = PCA(n_components=8)
X_new = pca_new.fit_transform(X)
print(X_new)

print(wine_data['quality'].value_counts())
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=.25, random_state=0)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42)

#sovracampionamento casuale
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

print("Class 1 numbers: " , len(y_train_res[y_train_res=="bad"]))
print("Class 2 numbers: " , len(y_train_res[y_train_res=="good"]))


print("Shape of X_train: ",X_train_res.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",y_train_res.shape)
print("Shape of y_test",y_test.shape)


# Support Vector Machines
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf', 'sigmoid']}

grid_svm = GridSearchCV(SVC(), param_grid=param_grid, cv=5, refit = True, verbose = False) 
grid_svm.fit(X_train_res, y_train_res) 
print("best_params", grid_svm.best_params_) 
print("best_estimator", grid_svm.best_estimator_) 
plot_results(grid_svm, X_test, y_test,'Support Vector Machines')


#Random Forest
rf=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [100, 250, 500],
    'max_features': ['auto', None, 'log2'],
    'criterion' :['gini', 'entropy']
}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5, refit = True, verbose = False)
grid_rf.fit(X_train_res, y_train_res)

print("best_params", grid_rf.best_params_) 
print("best_estimator", grid_rf.best_estimator_) 
plot_results(grid_rf, X_test, y_test,'Random Forest')

#K-nearest neighbors
knn = KNeighborsClassifier()
param_grid = {'n_neighbors':[1,4,5,6,7,8],
              'leaf_size':[1,3,5,10],
              'algorithm':['auto', 'kd_tree'],
              'n_jobs':[-1]}

grid_knn = GridSearchCV(knn, param_grid=param_grid)
grid_knn.fit(X_train_res ,y_train_res)

print("best_params", grid_knn.best_params_) 
print("best_estimator", grid_knn.best_estimator_) 
plot_results(grid_knn, X_test, y_test,'K-nearest neighbors')

# AdaBoost

Ada = AdaBoostClassifier(random_state=1)
Ada.fit(X_train_res, y_train_res)
plot_results(Ada, X_test, y_test,'AdaBoost')


#Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train_res, y_train_res)
plot_results(gaussian, X_test, y_test,'Gaussian Naive Bayes')

