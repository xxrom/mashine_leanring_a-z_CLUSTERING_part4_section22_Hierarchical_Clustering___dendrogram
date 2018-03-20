# Hierarchical Clustering
# работает хуже чем kmeans на больших наборах данных
# класторизация используется если мы не знаем какие есть группы
# такая же кластеризация как и kmeans, только мы запускаем его один раз
# смотрим дендрограмму и по набольшему вертикальному отрезку можно понять
# где нужно резать все остальные вертикальные линии
# вертикальный отрезок нужно брать максимальный самый без пересечений
# с горизонтальными линиями

# алгоритм класторизации очень простой:
# берем количество кластеров равное N (количество точек)
# потом объединяем две самые близкие точки(кластеры) из всех в один кластер
# потом ищем дальше самые ближайшие кластеры между собой (4 типа расстояний)
# и опять объединяем и так до тех пор пока не получится только один кластер

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values # income, spend rate [0-100]

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(
    sch.linkage(
        X,
        method = 'ward' # ??? не особо понял что это (l144)
      )
  )
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show() # дендрограмма, рисуем и находим, что 5 - количество кластеров

# Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering( # initicalization
    n_clusters = 5, # количество кластеров
    affinity = 'euclidean',# расстояние какое
    linkage = 'ward' # минимизация вариантов каждого кластера
  )
y_hc = hc.fit_predict(X) # predict same as k-Means

# Visualising the clusters same as in kmeans without centroids
plt.scatter(
    X[y_hc == 0, 0], # указали условие выборки X для 1 кластера (0)
    X[y_hc == 0, 1], # указали условие выборки Y для 1 кластера (0)
    s = 100, # вес кисти
    c = 'red',
    label = 'Careful'
  )
plt.scatter(
    X[y_hc == 1, 0], # указали условие выборки X для 2 кластера (1)
    X[y_hc == 1, 1], # указали условие выборки Y для 2 кластера (1)
    s = 100, # размер кластера
    c = 'blue',
    label = 'Standard'
  )
plt.scatter(
    X[y_hc == 2, 0], # указали условие выборки X для 3 кластера (2)
    X[y_hc == 2, 1], # указали условие выборки Y для 3 кластера (2)
    s = 100, # вес кисти
    c = 'green',
    label = 'Target'
  )
plt.scatter(
    X[y_hc == 3, 0], # указали условие выборки X для 4 кластера (3)
    X[y_hc == 3, 1], # указали условие выборки Y для 4 кластера (3)
    s = 100, # вес кисти
    c = 'cyan',
    label = 'Careless'
  )
plt.scatter(
    X[y_hc == 4, 0], # указали условие выборки X для 5 кластера (4)
    X[y_hc == 4, 1], # указали условие выборки Y для 5 кластера (4)
    s = 100, # вес кисти
    c = 'magenta',
    label = 'Sensible'
  )
plt.title('Clusters of clients')
plt.xlabel('Annual Incone (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()




















