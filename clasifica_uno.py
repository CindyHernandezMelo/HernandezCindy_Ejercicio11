import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

numero = 1

clasificador_train = y_train==numero
clasificador_test = y_test==numero

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-abs(valores))
valores = valores[ii]
vectores = vectores[:,ii]


F1_score_unos = np.zeros([38, 2])
F1_score_otros = np.zeros([38, 2])

clf_unos = LinearDiscriminantAnalysis()
clf_otros = LinearDiscriminantAnalysis()


    
for n_vectores in range(3,41):
       
    x_train_PCA =  x_train @ vectores[:,:n_vectores]
    x_test_PCA =   x_test @ vectores[:,:n_vectores]
    
    clf_unos.fit(x_train_PCA, clasificador_train)
    clf_otros.fit(x_train_PCA, ~clasificador_train)

    F1_score_unos[n_vectores - 3, 0] = f1_score(clasificador_train, clf_unos.predict(x_train_PCA))
    F1_score_unos[n_vectores - 3, 1] = f1_score(clasificador_test, clf_unos.predict(x_test_PCA))
    
    F1_score_otros[n_vectores - 3, 0] = f1_score(~clasificador_train, clf_otros.predict(x_train_PCA))
    F1_score_otros[n_vectores - 3, 1] = f1_score(~clasificador_test, clf_otros.predict(x_test_PCA))
    
    
    
plt.figure(figsize=(10,5))

x_axis = np.arange(3,41)

plt.subplot(121)
plt.scatter(x_axis, F1_score_unos[:,0], label = 'Train')
plt.scatter(x_axis, F1_score_unos[:,1], label = 'Test')
plt.title('Clasificacion Unos')
plt.xlabel('Numero de componentes PCA')
plt.ylabel('F1 Score')
plt.legend(loc = 'lower right')

plt.subplot(122)
plt.scatter(x_axis, F1_score_otros[:,0], label = 'Train')
plt.scatter(x_axis, F1_score_otros[:,1], label = 'Test')
plt.title('Clasificacion Otros')
plt.xlabel('Numero de componentes PCA')
plt.ylabel('F1 Score')
plt.legend(loc = 'lower right')

# Bono

# derivada

dF1 = (F1_score_unos[2:,1]-F1_score_unos[0:-2,1])/(2)
index_dF1 = np.argsort(dF1)
plt.subplot(121)
plt.axvline(x=index_dF1[1], c = 'r')

plt.subplot(122)
plt.axvline(x=index_dF1[1], c = 'r')

plt.savefig('F1_score_LinearDiscriminantAnalysis.png')


