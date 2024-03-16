# KNN for mushrooms data base

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import utils

# CARGAR Y DAR FORMATO A BASE DE DATOS

raw_data = pd.read_csv(r"C:\Users\edd_3\OneDrive\Escritorio\Coursera\data_analysis_python\semana_1\mushrooms.csv")

datas = pd.DataFrame({"class": raw_data["class"],
                     "cap-shape": raw_data["cap-shape"],
                     "cap-surface": raw_data["cap-surface"],
                     "cap-color": raw_data["cap-color"],
                     "habitat": raw_data["habitat"]
                    })


#print(data)
#print(data.info())

# Convert data in letter to numbers 

from sklearn.preprocessing import LabelEncoder

X = datas[datas.columns[1:5]]
y = datas[datas.columns[0]]

#print(y)
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#le_2 = LabelEncoder()
for i in X:
    X[i] = le.fit_transform(X[i])
    #print(X[i])

#print(X)


# VISUALIZATION 

visual = sns.relplot(data = datas, x = 'habitat', y = "cap-shape", hue = 'class')
plt.show()

# DIVISIÓN PARA TRAIN Y TEST

from sklearn.model_selection import train_test_split

#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#print(X_train[:5])
#print(X_test[:5])
#print(y_train[:5])
#print(y_test[:5])

# SCALING, no se hará en este caso porque son letras, no números, jeje
"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train[:5])
print(X_test[:5])
"""

# TRAINING THE MODEL

from sklearn.neighbors import KNeighborsClassifier

k = 7
classifer = KNeighborsClassifier(n_neighbors= k).fit(X_train, y_train)

y_pred = classifer.predict(X_test)

# TESTING AND MEASURING MODEL

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

# FINING THE BEST K 

def knn_tuning(k):
    classifer = KNeighborsClassifier(n_neighbors = k)
    classifer.fit(X_train, y_train)
    y_pred = classifer.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

knn_results = pd.DataFrame({'k':np.arange(1, 100, 5)})
#print(knn_results)

knn_results['Accuracy'] = knn_results['k'].apply(knn_tuning)
#print(knn_results)

# OPTIMIZE WEIGHTS

def knn_tuning_uniform(k):
  classifier = KNeighborsClassifier(n_neighbors = k, weights= 'uniform')
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  accuracy = accuracy_score(y_test,y_pred)
  return accuracy


def knn_tuning_distance(k):
  classifier = KNeighborsClassifier(n_neighbors = k, weights= 'distance')
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  accuracy = accuracy_score(y_test,y_pred)
  return accuracy

knn_results['Uniform'] = knn_results['k'].apply(knn_tuning_uniform)
knn_results['Distance'] = knn_results['k'].apply(knn_tuning_distance)
print(knn_results)
