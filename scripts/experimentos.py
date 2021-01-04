# ################################################################
# Universidade Federal de Sao Carlos (UFSCAR)
# Aprendizado de Maquina - 2020
# Projeto Final

# Aluno: Eduardo Garcia do Nascimento
# RA/CPF: 22008732800
# ################################################################

# Arquivo com todas as funcoes e codigos referentes aos experimentos

# Classificadores utilizados para otimização de hiperparâmetros
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

import time
import numpy as np


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Avalia o modelo com base nos dados de treinamento e teste fornecidos.

    Parâmetros
    ----------------
    model: estimator
    Modelo a ser avaliado

    X_train : array
    Array contendo os atributos da base de treino.

    y_train : array
    Array contendo as classes da base de treino.

    X_test : array
    Array contendo os atributos da base de teste.

    y_test : array
    Array contendo as classes da base de teste.

    Retornos
    ----------------
    results: dicionário
    Dicionário com as pontuações do modelo e base fornecidos.

    """
    results = {}
    start = time.time()
    clf = model.fit(X_train, y_train)
    end = time.time()
    train_time = end-start
    y_pred = clf.predict(X_train)
    results['bal_acc_train'] = metrics.balanced_accuracy_score(y_train, y_pred)
    start = time.time()
    y_pred = clf.predict(X_test)
    np.savetxt("y_pred.csv", y_pred, delimiter=",")
    end = time.time()
    predict_time = end-start
    results['bal_acc_test'] = metrics.balanced_accuracy_score(y_test, y_pred)
    results['f1_weighted'] = metrics.f1_score(y_test, y_pred, average='weighted')
    results['f1_micro'] = metrics.f1_score(y_test, y_pred, average='micro')
    results['precision'] = metrics.precision_score(y_test, y_pred)
    results['recall'] = metrics.recall_score(y_test, y_pred)
    y_proba = clf.predict_proba(X_test)
    np.savetxt("y_proba.csv", y_proba, delimiter=",")
    np.savetxt("y_test.csv", y_test, delimiter=",")
    results['mcc'] = metrics.matthews_corrcoef(y_test, y_pred)
    results['roc_auc'] = metrics.roc_auc_score(y_test, y_proba[:,1], average='micro')
    results['TP'] = confusion_matrix(y_test, y_pred)[0][0]
    results['TN'] = confusion_matrix(y_test, y_pred)[1][1]
    results['train_time'] = train_time
    results['predict_time'] = predict_time

    return results

def cross_classifier(model, X1_train, y1_train, X1_test, y1_test,
                            X2_train, y2_train, X2_test, y2_test,
                            X3_train, y3_train, X3_test, y3_test):
    
    results = {}
    clf1 = model.fit(X1_train, y1_train)
    y1_pred = clf1.predict(X1_train)
    y1_pred = clf1.predict(X1_test)
    y1_proba = clf1.predict_proba(X1_test)

    clf2 = model.fit(X2_train, y2_train)
    y2_pred = clf2.predict(X2_train)
    y2_pred = clf2.predict(X2_test)
    y2_proba = clf2.predict_proba(X2_test)

    clf3 = model.fit(X3_train, y3_train)
    y3_pred = clf3.predict(X3_train)
    y3_pred = clf3.predict(X3_test)
    y3_proba = clf3.predict_proba(X3_test)

    y_pred = np.empty(y1_pred.shape)
    y_proba = np.empty(y1_proba.shape)

    for i in range(y1_proba.shape[0]):
        if( (abs(y1_proba[i][0]-y1_proba[i][1]) >= abs(y2_proba[i][0]-y2_proba[i][1])) and (abs(y1_proba[i][0]-y1_proba[i][1]) >= abs(y3_proba[i][0]-y3_proba[i][1])) ):
            y_proba[i] = y1_proba[i]
            y_pred = y1_pred
        if( (abs(y2_proba[i][0]-y2_proba[i][1]) >= abs(y1_proba[i][0]-y1_proba[i][1])) and (abs(y2_proba[i][0]-y2_proba[i][1]) >= abs(y3_proba[i][0]-y3_proba[i][1])) ):
            y_proba[i] = y2_proba[i]
            y_pred = y2_pred
        if( (abs(y3_proba[i][0]-y3_proba[i][1]) >= abs(y2_proba[i][0]-y2_proba[i][1])) and (abs(y3_proba[i][0]-y3_proba[i][1]) >= abs(y1_proba[i][0]-y1_proba[i][1])) ):
            y_proba[i] = y3_proba[i]
            y_pred = y3_pred

    np.savetxt("y_pred.csv", y_pred, delimiter=",")
    
    #results['bal_acc_train'] = metrics.balanced_accuracy_score(y1_train, y_pred)
    results['bal_acc_test'] = metrics.balanced_accuracy_score(y1_test, y_pred)
    results['f1_weighted'] = metrics.f1_score(y1_test, y_pred, average='weighted')
    results['f1_micro'] = metrics.f1_score(y1_test, y_pred, average='micro')
    results['precision'] = metrics.precision_score(y1_test, y_pred)
    results['recall'] = metrics.recall_score(y1_test, y_pred)
    
    np.savetxt("y_proba.csv", y_proba, delimiter=",")
    np.savetxt("y_test.csv", y1_test, delimiter=",")
    results['mcc'] = metrics.matthews_corrcoef(y1_test, y_pred)
    results['roc_auc'] = metrics.roc_auc_score(y1_test, y_proba[:,1], average='micro')
    results['TP'] = confusion_matrix(y1_test, y_pred)[0][0]
    results['TN'] = confusion_matrix(y1_test, y_pred)[1][1]
    #results['train_time'] = train_time
    #results['predict_time'] = predict_time

    return results

def cross_evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Avalia o modelo fazendo validação cruzada com base nos dados de treinamento e teste fornecidos.

    Parâmetros
    ----------------
    model: estimator
    Modelo a ser avaliado

    X_train : array
    Array contendo os atributos da base de treino.

    y_train : array
    Array contendo as classes da base de treino.

    X_test : array
    Array contendo os atributos da base de teste.

    y_test : array
    Array contendo as classes da base de teste.

    Retornos
    ----------------
    results: dicionário
    Dicionário com as pontuações do modelo e base fornecidos.

    Observações
    ----------------
    Esta função tornou-se obsoleta neste atrabalho a partir do momento que a aprendizado semi-supervisionado
    foi introduzido. O mesmo insere amostras dentro da base que tornam o resultado da avaliação não confiável
    já que são amostras bem conhecidas geradas a partir do pŕoprio classificador.

    """
    clf = model.fit(X_train, y_train)
    y_pred = cross_val_predict(clf,X_train, y_train)
    print("Acurácia Treino:",metrics.accuracy_score(y_train, y_pred))
    y_pred = cross_val_predict(clf,X_test, y_test)
    print("Acurácia Teste:",metrics.accuracy_score(y_test, y_pred))
    y_proba = cross_val_predict(clf,X_test, y_test, method='predict_proba')
    print("AUC score:", metrics.roc_auc_score(y_test, y_proba[:,1]))
    print("F1 score:", metrics.f1_score(y_test, y_pred))


    return y_pred

def find_best_knn(X, y, scores):
    """
    Através da busca em grid encontra os melhores hiperparâmetros a serem utilizados
    para o classificador KNN.

    Parâmetros
    ----------------
    X : array
    Array contendo os atributos da base de treino.

    y : array
    Array contendo as classes da base de treino.

    scores : array
    Pontuações a serem consideradas para definição dos melhores hiperparâmetros.

    """
    print('Buscando os melhores parâmetros para o KNN:')

    cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    tuned_parameters = { 'n_neighbors': range(1,30),
                         'leaf_size':range(1,9,2),
                         'algorithm':['auto', 'kd_tree'],
                         'weights': ['distance', 'uniform'],
                       }

    for score in scores:
        clf = GridSearchCV(
                KNeighborsClassifier(weights='distance'), tuned_parameters, scoring=score, cv=cv
            )
        clf.fit(X, y)

        print('Os melhores parâmetros encontrados para a pontuação %s foram:' % score)
        print(clf.best_params_)


def find_best_svm(X, y, scores):
    """
    Através da busca em grid encontra os melhores hiperparâmetros a serem utilizados
    para o classificador máquinas de vetores de suporte.

    Parâmetros
    ----------------
    X : array
    Array contendo os atributos da base de treino.

    y : array
    Array contendo as classes da base de treino.

    scores : array
    Pontuações a serem consideradas para definição dos melhores hiperparâmetros.

    """

    cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['scale', 1e-3, 1e-4], 'C': [1, 10, 100, 200, 400, 600, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 200, 400, 600, 1000]},
                        {'kernel': ['poly'], 'C': [1, 10, 100, 200, 400, 600, 1000], 'degree': [2,3,4,5,6], 'gamma': ['scale', 1e-3, 1e-4]}]

    
    for score in scores:
        clf = GridSearchCV(
                svm.SVC(class_weight='balanced', random_state=1), tuned_parameters, scoring=score, cv=cv
            )
        clf.fit(X, y)

        print('Os melhores parâmetros encontrados para a pontuação %s foram:' % score)
        print(clf.best_params_)

def find_best_rfc(X, y, scores):
    """
    Através da busca em grid encontra os melhores hiperparâmetros a serem utilizados
    para o classificador florestas aleatórias.

    Parâmetros
    ----------------
    X : array
    Array contendo os atributos da base de treino.

    y : array
    Array contendo as classes da base de treino.

    scores : array
    Pontuações a serem consideradas para definição dos melhores hiperparâmetros.

    """
    print('Buscando os melhores parâmetros para as florestas aleatórias:')

    cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    tuned_parameters = { 'n_estimators': [100, 200, 500],
                         'max_features': ['auto', 'sqrt', 'log2'],
                         'max_depth' : [None, 2,3,4,5,6,7,8],
                         'criterion' :['gini', 'entropy']}

    for score in scores:
        clf = GridSearchCV(
                RandomForestClassifier(class_weight='balanced'), tuned_parameters, scoring=score, cv=cv
            )
        clf.fit(X, y)

        print('Os melhores parâmetros encontrados para a pontuação %s foram:' % score)
        print(clf.best_params_)

def find_best_mlp(X, y, scores):
    """
    Através da busca em grid encontra os melhores hiperparâmetros a serem utilizados
    para o classificador utilizando redes neurais.

    Parâmetros
    ----------------
    X : array
    Array contendo os atributos da base de treino.

    y : array
    Array contendo as classes da base de treino.

    scores : array
    Pontuações a serem consideradas para definição dos melhores hiperparâmetros.

    """
    print('Buscando os melhores parâmetros para Multi Layer Perceptron (Neural Networks):')

    cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    tuned_parameters = {'solver': ['sgd', 'adam'],
                        'learning_rate': ["constant", "invscaling", "adaptive"],
                        'activation': ["logistic", "relu", "Tanh"],
                        'alpha': 10.0 ** -np.arange(1, 10),
                        'hidden_layer_sizes': [(100,), (200,), (400,), (600,), (800,)]}

    for score in scores:
        clf = GridSearchCV(
            MLPClassifier(), tuned_parameters, scoring=score, cv=cv, verbose=10
        )
        clf.fit(X, y)

        print('Os melhores parâmetros encontrados para a pontuação %s foram:' % score)
        print(clf.best_params_)

    return clf