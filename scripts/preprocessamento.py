# ################################################################
# Universidade Federal de Sao Carlos (UFSCAR)
# Aprendizado de Maquina - 2020
# Projeto Final

# Aluno: Eduardo Garcia do Nascimento
# RA/CPF: 22008732800
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

import numpy as np 
import pandas as pd
from sklearn.utils import resample
from scipy import stats
from sklearn import metrics

def remove_outliers(X,y):
    """
    Remove os outliers da base fornecida

    Esta função remove os outliers separando por classes pra garantir que amostras
    válidas de uma classe minoritária não sejam removidos indevidamente.

    Parâmetros
    ----------------
    X : array
    Array contendo os atributos da base de dados.

    y : array
    Array contendo as classes da base de dados fornecida.
    """
    xdf = pd.DataFrame(data=X)
    ydf = pd.DataFrame(data=y, columns=['class'])
    zdf = pd.concat([xdf,ydf], axis=1)
    xdf_o = zdf[zdf['class']==1].copy()
    xdf_m = zdf[zdf['class']==-1].copy()
    
    z = np.abs(stats.zscore(xdf_o.drop('class', axis=1)))
    z = np.nan_to_num(z)
    samples_removed_pct = 100 * (xdf_o.shape[0] - np.sum((z < 3).all(axis=1)))/xdf_o.shape[0]
    print('Amostras positivas (phishing) mantidas: %d de %d' % (np.sum((z < 3).all(axis=1)), xdf_o.shape[0]))
    print('Percentual de outliers removidos: %.1f%%' % samples_removed_pct)
    xdf_o = xdf_o[(z < 3).all(axis=1)]
    
    z = np.abs(stats.zscore(xdf_m.drop('class', axis=1)))
    z = np.nan_to_num(z)
    samples_removed_pct = 100 * (xdf_m.shape[0] - np.sum((z < 3).all(axis=1)))/xdf_m.shape[0]
    print('Amostras negativas (HAM) mantidas: %d de %d' % (np.sum((z < 3).all(axis=1)), xdf_m.shape[0]))
    print('Percentual de outliers removidos: %.1f%%' % samples_removed_pct)
    xdf_m = xdf_m[(z < 3).all(axis=1)]
    
    totaldf = pd.concat([xdf_o,xdf_m])
    y = totaldf['class'].values
    X = totaldf.drop('class', axis=1).values

    return X,y


def oversample(X,y, times=None):
    """
    Aplica a estratégia de oversample na base fornecida

    Uma das estratégias possíveis para ajustamento de bases desbalanceadas é a de sobreamostragem.
    Apesar de balancear as classes é comum que surja um overfitting do classificador quando a
    mesma é aplicada sobre a base de dados.

    Parâmetros
    ----------------
    X : array
    Array contendo os atributos da base de dados.

    y : array
    Array contendo as classes da base de dados fornecida.
    """
    print('Balaceamento antes da SOBREamostragem', np.sum(y==-1),np.sum(y==1))

    xdf = pd.DataFrame(data=X)
    ydf = pd.DataFrame(data=y, columns=['class'])
    zdf = pd.concat([xdf,ydf], axis=1)
    xdf_o = zdf[zdf['class']==1].copy()
    xdf_m = zdf[zdf['class']==-1].copy()
    


    if xdf_m.shape[0] > xdf_o.shape[0]:
        if times is None:
            xdf_o = resample(xdf_o, 
                            replace=True,     # sample with replacement
                            n_samples=xdf_m.shape[0],    # to match majority class
                            random_state=123) # reproducible results
        else:
            xdf_o = resample(xdf_o, 
                            replace=True,     # sample with replacement
                            n_samples=xdf_o.shape[0] * times,    # to match majority class
                            random_state=123) # reproducible results
    
    if xdf_m.shape[0] < xdf_o.shape[0]:
        if times is None:
            xdf_m = resample(xdf_m, 
                            replace=True,     # sample with replacement
                            n_samples=xdf_o.shape[0],    # to match majority class
                            random_state=123) # reproducible results
        else:
            xdf_m = resample(xdf_m, 
                            replace=True,     # sample with replacement
                            n_samples=xdf_m.shape[0] * times,    # to match majority class
                            random_state=123) # reproducible results
        

    totaldf = pd.concat([xdf_o,xdf_m])
    y = totaldf['class'].to_numpy() 
    X = totaldf.drop('class', axis=1).to_numpy()

    print('Balaceamento após da SOBREamostragem', np.sum(y==-1),np.sum(y==1))

    return X,y


def downsample(X,y):
    """
    Aplica a estratégia de downsample na base fornecida

    Uma das estratégias possíveis para ajustamento de bases desbalanceadas é a de subamostragem.
    Neste caso apesar de balancear as classes é comum que seja causado o underfitting do classificador
    já que estamos retirando aleatóriamente amostras que podem ser importantes para generalização
    das amostras de teste ou validação.

    Parâmetros
    ----------------
    X : array
    Array contendo os atributos da base de dados.

    y : array
    Array contendo as classes da base de dados fornecida.
    """
    print('Balaceamento antes da subamostragem', np.sum(y==-1),np.sum(y==1))
    
    xdf = pd.DataFrame(data=X)
    ydf = pd.DataFrame(data=y, columns=['class'])
    zdf = pd.concat([xdf,ydf], axis=1)
    xdf_o = zdf[zdf['class']==1].copy()
    xdf_m = zdf[zdf['class']==-1].copy()


    if xdf_o.shape[0] > xdf_m.shape[0]:
        xdf_o = resample(xdf_o, 
                        replace=False,     # sample with replacement
                        n_samples=xdf_m.shape[0],    # to match majority class
                        random_state=123) # reproducible results
    if xdf_o.shape[0] < xdf_m.shape[0]:
        xdf_m = resample(xdf_m, 
                        replace=False,     # sample with replacement
                        n_samples=xdf_o.shape[0],    # to match majority class
                        random_state=123) # reproducible results
                    

    totaldf = pd.concat([xdf_o,xdf_m])
    y = totaldf['class'].to_numpy() 
    X = totaldf.drop('class', axis=1).to_numpy()
    print('Balaceamento após da subamostragem', np.sum(y==-1),np.sum(y==1))

    return X,y


def add_samples(model, X, y, samples):
    """
    Coleta mais amostras para a base de dados

    Aplicando uma estratégia análoga ao aprendizado semi-supervisionado, adiciona mais amostras
    à base existente utilizando as amostras não classificadas de uma base fornecida.

    Parâmetros
    ----------------
    model: estimator
    Classificador utilizado para coleta de mais amostras.

    X : array
    Array contendo os atributos da base de dados.

    y : array
    Array contendo as classes da base de dados fornecida.

    samples: array
    Array com as amostras não classificadas a serem integradas à base de treino

    Retornos
    ----------------
    X_semi: array
    Array com os atributos e dados das novas amostras encontradas com maior grau de confiabilidade.

    y_semi: array
    Array com as classes das novas amostras encontradas com maior grau de confiabilidade.

    proba_mask: array
    Array com a máscara das amostras que foram classificadas com alto grau de confiabilidade.

    """
    clf = model.fit(X, y)
    y_semi = clf.predict(samples)
    y_probas = clf.predict_proba(samples)
    print(y_probas)
    # O nível de confiança está aqui fixo para >= 80% para ambas as classes
    proba_mask = (y_probas[:,0] < 0.001) | (y_probas[:,0] > 0.999)
    y_semi = y_semi[proba_mask]
    X_semi = samples[proba_mask]
    
    return X_semi, y_semi, proba_mask
