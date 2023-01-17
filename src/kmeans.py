import os
import sys
import csv
import numpy
import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, normalize
from sklearn.decomposition import PCA

def modelAndDraw(df_values):
    model = KMeans(n_init='auto', n_clusters=3)
    model = model.fit(df_values)
    predictions = model.predict(df_values)

    plt.figure()
    plt.scatter(df_values[:, 0], df_values[:, 1], c=predictions)
    plt.show()

def pca(df_values):
    pca_2 = PCA(n_components=2)
    pca_2_result = pca_2.fit_transform(df_values)
    print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))

    # >> Explained variation per principal component: [0.36198848 0.1920749 ]

    print('Cumulative variance explained by 2 principal components: {:.2%}'.format(numpy.sum(pca_2.explained_variance_ratio_)))

    print("\n*************** Most important features *************************")
    print('As per PC 1:\n', (pca_2.components_)) 
    print("\n******************************************************************")
    return pca_2_result

df = pandas.read_csv(sys.argv[1])
df = df.fillna(0)

X = df[[ "p90_len", "mt_mo_ratio"]].values
X_norm = normalize(X)
X_scale = scale(X)
X_pca = pca(X_scale)

#modelAndDraw(X_norm)
#modelAndDraw(X_scale)
modelAndDraw(X_pca)


