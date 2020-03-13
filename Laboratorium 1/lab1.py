import numpy as np
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

dataset = load_breast_cancer()
X_data, y_data = dataset.data, dataset.target #podział danych na macierz cech i wektor klas

data_normalized=(X_data-np.tile(np.min(X_data,axis=0), (569,1))) / (np.tile(np.max(X_data,axis=0)-np.min(X_data,axis=0), (569,1)))

#wyświetlenie danych w 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_data[y_data==0,0],X_data[y_data==0,1],X_data[y_data==0,2], c='b', marker='x', label=dataset.target_names[0])
ax.scatter(X_data[y_data==1,0],X_data[y_data==1,1],X_data[y_data==1,2], c='g', marker='o', label=dataset.target_names[1])

ax.set_xlabel(dataset.feature_names[0])
ax.set_ylabel(dataset.feature_names[1])

plt.show()

#wyświetlenie danych z redukcją jednego wymiaru
model = TSNE(learning_rate=100)
transformed = model.fit_transform(X_data)

xs=transformed[:,0]
ys=transformed[:,1]

plt.figure()
plt.scatter(xs,ys,c=y_data)
plt.show()

#wyszukanie trzech cech ręcznie
# próba znalezienia mało skorelowanych cech
a= np.arange(0,30,1)
plt.plot(X_data[1],a, 'ro')
plt.plot(X_data[140],a, 'bo')

b = np.sort(X_data[1])
b2 = np.sort(X_data[140])
b3 = X_data[140]
c = X_data[1]
max30 = b[-3:]
max31 = b2[-3:]
wynik1 = [] 
for znak in max30:
    wynik1.append(np.where(c == znak))
for znak1 in max31:
    wynik1.append(np.where(b3 == znak1))     
wynik1

#wybierz tylko wartości o indexie 22,3,23 
a = X_data[:,3] 
b = X_data[:,23]
c = X_data[:,22]
X_train = np.column_stack((a,b,c))
np.shape(X_train)
# teraz moje dane to X_train i y_data pozostaje bez zmian
#wyświetlenie danych w 3D (tylko 3 cechy)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_train[y_data==0,0],X_train[y_data==0,1],X_train[y_data==0,2], c='b', marker='x', label=dataset.target_names[0])
ax.scatter(X_train[y_data==1,0],X_train[y_data==1,1],X_train[y_data==1,2], c='g', marker='o', label=dataset.target_names[1])

ax.set_xlabel(dataset.feature_names[0])
ax.set_ylabel(dataset.feature_names[1])

plt.show()

model = TSNE(learning_rate=100)
transformed = model.fit_transform(X_train)

xs=transformed[:,0]
ys=transformed[:,1]

plt.figure()
plt.scatter(xs,ys,c=y_data)
plt.show()
