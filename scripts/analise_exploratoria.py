# ################################################################
# Universidade Federal de Sao Carlos (UFSCAR)
# Aprendizado de Maquina - 2020
# Projeto Final

# Aluno: Eduardo Garcia do Nascimento
# RA/CPF: 22008732800
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def printPCA(X, y):
    """
    Plota a análise de componentes principais do conjunto 

    Aplica a análise de componentes principais no conjunto fornecido reduzindo
    o mesmo para somente duas dimensões e em seguida coloca o resultado num diagrama
    de dispersão separando os pontos pelas classes unitárias fornecidas.

    Parâmetros
    ----------------
    X : array
    Array contendo os atributos da base de dados.

    y : array
    Array contendo as classes da base de dados fornecida.
    """
    pca = PCA(2) 
    projected = pca.fit_transform(X)
    colors = ['r', 'b']
    markers = ['x', 'x']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X[y==l, 0], 
                    X[y==l, 1], 
                    c=c, label=l, marker=m)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower right')
    plt.show()

def printJointPlot(X, y):
    pca = PCA(2)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(X)
    data = pd.DataFrame(data=projected, columns=['a', 'b'])
    data['classe'] = y

    if int(sns.__version__[2:4]) < 11:
        print('Para uma melhor experiência, atualize a seaborn para a versão 11 ou superior')
        sns.jointplot(data=data, x="a", y="b")
    else:
        sns.jointplot(data=data, x="a", y="b", kind="kde", palette='Spectral', hue='classe')