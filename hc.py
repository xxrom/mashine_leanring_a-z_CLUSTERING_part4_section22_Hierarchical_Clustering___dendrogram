# Hierarchical Clustering
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