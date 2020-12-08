import numpy as np 
import pandas as pd
from sklearn.utils import resample
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
import os
import time


def remove_outliersandnan_perclass(X, y):
    zdf = pd.concat([X,y], axis=1)
    xdf_o = zdf[zdf['classe']==1].copy()
    xdf_m = zdf[zdf['classe']==-1].copy()
    xdf_r = zdf[(zdf['classe']!=-1)&(zdf['classe']!=1)].copy()

    Q1 = xdf_o.quantile(0.25)
    Q3 = xdf_o.quantile(0.75)
    IQR = Q3 - Q1
    mask = (xdf_o < (Q1 - 1.5 * IQR)) | (xdf_o > (Q3 + 1.5 * IQR))
    xdf_o[mask] = np.nan
    print('Preenchendo nan com estimativas....................................')
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    xdf_o.loc[:,:] = imp_mean.fit_transform(xdf_o)

    Q1 = xdf_m.quantile(0.25)
    Q3 = xdf_m.quantile(0.75)
    IQR = Q3 - Q1
    mask = (xdf_m < (Q1 - 1.5 * IQR)) | (xdf_m > (Q3 + 1.5 * IQR))
    xdf_m[mask] = np.nan
    print('Preenchendo nan com estimativas....................................')
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    xdf_m.loc[:,:] = imp_mean.fit_transform(xdf_m)

    Q1 = xdf_r.quantile(0.25)
    Q3 = xdf_r.quantile(0.75)
    IQR = Q3 - Q1
    mask = (xdf_r < (Q1 - 1.5 * IQR)) | (xdf_r > (Q3 + 1.5 * IQR))
    xdf_r[mask] = np.nan
    print('Preenchendo nan com estimativas....................................')
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    xdf_r.loc[:,:] = imp_mean.fit_transform(xdf_r)
    
   
    totaldf = pd.concat([xdf_o,xdf_m, xdf_r])
    y = totaldf['classe']
    X = totaldf.drop('classe', axis=1)

    return X,y
    
def remove_outliersandnan(X):
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    mask = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
    print('Outliers encontrados e removidos: %d' % np.count_nonzero(mask == True))
    X.loc[mask] = np.nan
    X = X.fillna(X.median())
    #z = np.abs(stats.zscore(X))
    #z = np.nan_to_num(z)
    #print('Amostras =1 mantidas: %d de %d' % (np.sum((z < 3).all(axis=1)), X.shape[0]))
    #X = X[(z < 3).all(axis=1)]

    return X

def remove_outliers(X,y):
    xdf = pd.DataFrame(data=X)
    ydf = pd.DataFrame(data=y, columns=['class'])
    zdf = pd.concat([xdf,ydf], axis=1)
    xdf_o = zdf[zdf['class']==1].copy()
    xdf_m = zdf[zdf['class']==-1].copy()
    
    z = np.abs(stats.zscore(xdf_o.drop('class', axis=1)))
    z = np.nan_to_num(z)
    print('Amostras =1 mantidas: %d de %d' % (np.sum((z < 3).all(axis=1)), xdf_o.shape[0]))
    xdf_o = xdf_o[(z < 3).all(axis=1)]
    
    z = np.abs(stats.zscore(xdf_m.drop('class', axis=1)))
    z = np.nan_to_num(z)
    print('Amostras =-1 mantidas: %d de %d' % (np.sum((z < 3).all(axis=1)), xdf_m.shape[0]))
    xdf_m = xdf_m[(z < 3).all(axis=1)]
    
    totaldf = pd.concat([xdf_o,xdf_m])
    y = totaldf['class'].values
    X = totaldf.drop('class', axis=1).values

    return X,y


def balance_classes_up(X,y, remove_outliers):
    xdf = pd.DataFrame(data=X)
    ydf = pd.DataFrame(data=y, columns=['class'])
    zdf = pd.concat([xdf,ydf], axis=1)
    xdf_o = zdf[zdf['class']==1].copy()
    xdf_m = zdf[zdf['class']==-1].copy()
    xdf_r = zdf[(zdf['class']!=-1) & zdf[zdf['class']!=1]].copy()
    
    xdf_o = xdf_o.replace([np.inf, -np.inf], np.nan)
    xdf_o = xdf_o.fillna(xdf_o.median())
    Q1 = xdf_o.quantile(0.25)
    Q3 = xdf_o.quantile(0.75)
    IQR = Q3 - Q1
    mask = (xdf_o < (Q1 - 1.5 * IQR)) | (xdf_o > (Q3 + 1.5 * IQR))
    print('Outliers encontrados e removidos: %d' % np.count_nonzero(mask == True))
    xdf_o[mask] = np.nan
    xdf_o = xdf_o.fillna(xdf_o.median())
    if remove_outliers == True:
        z = np.abs(stats.zscore(xdf_o))
        z = np.nan_to_num(z)
        print('Amostras =1 mantidas: %d de %d' % (np.sum((z < 3).all(axis=1)), xdf_o.shape[0]))
        xdf_o = xdf_o[(z < 3).all(axis=1)]

    xdf_o = xdf_o.replace([np.inf, -np.inf], np.nan)
    xdf_m = xdf_m.fillna(xdf_o.median())
    Q1 = xdf_m.quantile(0.25)
    Q3 = xdf_m.quantile(0.75)
    IQR = Q3 - Q1
    mask = (xdf_m < (Q1 - 1.5 * IQR)) | (xdf_m > (Q3 + 1.5 * IQR))
    print('Outliers encontrados e removidos: %d' %  np.count_nonzero(mask == True))
    xdf_m[mask] = np.nan
    xdf_m = xdf_m.fillna(xdf_m.median())
    if remove_outliers == True:
        z = np.abs(stats.zscore(xdf_m))
        z = np.nan_to_num(z)
        print('Amostras =-1 mantidas: %d de %d' % (np.sum((z < 3).all(axis=1)), xdf_m.shape[0]))
        xdf_m = xdf_m[(z < 3).all(axis=1)]
    #-------------------------------
    #xdf_o = resample(xdf_o, 
    #                replace=True,     # sample with replacement
    #                n_samples=xdf_m.shape[0],    # to match majority class
    #                random_state=123) # reproducible results

    totaldf = pd.concat([xdf_o,xdf_m, xdf_r])
    y = totaldf['class']
    X = totaldf.drop('class', axis=1)

    return X,y

def balance_classes_commom(X,y, remove_outliers):
    xdf = pd.DataFrame(data=X)
    xdf = xdf.replace([np.inf, -np.inf], np.nan)
    xdf = xdf.fillna(xdf.median())
    Q1 = xdf.quantile(0.25)
    Q3 = xdf.quantile(0.75)
    IQR = Q3 - Q1
    mask = (xdf < (Q1 - 1.5 * IQR)) | (xdf > (Q3 + 1.5 * IQR))
    print('Outliers encontrados e removidos: %d' % np.count_nonzero(mask == True))
    xdf[mask] = np.nan
    xdf = xdf.fillna(xdf.median())
    if remove_outliers == True:
        z = np.abs(stats.zscore(xdf))
        z = np.nan_to_num(z)
        print('Amostras =1 mantidas: %d de %d' % (np.sum((z < 3).all(axis=1)), xdf.shape[0]))
        xdf = xdf[(z < 3).all(axis=1)]



    ydf = pd.DataFrame(data=y, columns=['class'])
    zdf = pd.concat([xdf,ydf], axis=1)
    xdf_o = zdf[zdf['class']==1].copy()
    xdf_m = zdf[zdf['class']==-1].copy()
    xdf_r = zdf[(zdf['class']!=-1) & zdf[zdf['class']!=1]].copy()
    
    

    xdf_o = xdf_o.replace([np.inf, -np.inf], np.nan)
    xdf_m = xdf_m.fillna(xdf_o.median())
    Q1 = xdf_m.quantile(0.25)
    Q3 = xdf_m.quantile(0.75)
    IQR = Q3 - Q1
    mask = (xdf_m < (Q1 - 1.5 * IQR)) | (xdf_m > (Q3 + 1.5 * IQR))
    print('Outliers encontrados e removidos: %d' %  np.count_nonzero(mask == True))
    xdf_m[mask] = np.nan
    xdf_m = xdf_m.fillna(xdf_m.median())
    if remove_outliers == True:
        z = np.abs(stats.zscore(xdf_m))
        z = np.nan_to_num(z)
        print('Amostras =-1 mantidas: %d de %d' % (np.sum((z < 3).all(axis=1)), xdf_m.shape[0]))
        xdf_m = xdf_m[(z < 3).all(axis=1)]
    #-------------------------------
    #xdf_o = resample(xdf_o, 
    #                replace=True,     # sample with replacement
    #                n_samples=xdf_m.shape[0],    # to match majority class
    #                random_state=123) # reproducible results

    totaldf = pd.concat([xdf_o,xdf_m, xdf_r])
    y = totaldf['class']
    X = totaldf.drop('class', axis=1)

    return X,y


def balance_classes_down(X,y):
    xdf = pd.DataFrame(data=X)
    ydf = pd.DataFrame(data=y, columns=['class'])
    zdf = pd.concat([xdf,ydf], axis=1)
    xdf_o = zdf[zdf['class']==1].copy()
    xdf_m = zdf[zdf['class']==0].copy()
    
    Q1 = xdf_o.quantile(0.25)
    Q3 = xdf_o.quantile(0.75)
    IQR = Q3 - Q1
    mask = (xdf_o < (Q1 - 1.5 * IQR)) | (xdf_o > (Q3 + 1.5 * IQR))
    xdf_o[mask] = np.nan
    xdf_o = xdf_o.replace([np.inf, -np.inf], np.nan)
    xdf_o = xdf_o.fillna(xdf_o.mean())
    z = np.abs(stats.zscore(xdf_o))
    z = np.nan_to_num(z)
    xdf_o = xdf_o[(z < 3).all(axis=1)]
    
    Q1 = xdf_m.quantile(0.25)
    Q3 = xdf_m.quantile(0.75)
    IQR = Q3 - Q1
    mask = (xdf_m < (Q1 - 1.5 * IQR)) | (xdf_m > (Q3 + 1.5 * IQR))
    xdf_m[mask] = np.nan
    xdf_m = xdf_m.replace([np.inf, -np.inf], np.nan)
    xdf_m = xdf_m.fillna(xdf_m.mean())
    z = np.abs(stats.zscore(xdf_m))
    z = np.nan_to_num(z)
    xdf_m = xdf_m[(z < 3).all(axis=1)]
    xdf_m = resample(xdf_m, 
                    replace=False,     # sample with replacement
                    n_samples=xdf_o.shape[0],    # to match majority class
                    random_state=123) # reproducible results

    totaldf = pd.concat([xdf_o,xdf_m])
    y = totaldf['class'].to_numpy()
    X = totaldf.drop('class', axis=1).iloc[:,:].values

    return X,y

def clean_dataset(X):
    df = pd.DataFrame(data=X)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    df[mask] = np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean())

    return df.to_numpy()

def remove_lowvar(X):
    print('Removendo atributos com baixa variância....................................')
    df = pd.DataFrame(data=X)
    variance_mask = VarianceThreshold().fit(X).get_support()
    X = X[:,variance_mask]
    print('Um total de %d atributos foi removido' % np.sum(~variance_mask))

    return X.to_numpy(), variance_mask

def remove_correlated(X, threshold):
    df = pd.DataFrame(data=X)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop features 
    X = df.drop(to_drop, axis=1).values
    return X, to_drop


def balance_classes(X,y):
    xdf = pd.DataFrame(data=X)
    ydf = pd.DataFrame(data=y, columns=['class'])
    zdf = pd.concat([xdf,ydf], axis=1)
    xdf_o = zdf[zdf['class']==1].copy()
    xdf_m = zdf[zdf['class']==-1].copy()
    
    xdf_o = resample(xdf_o, 
                    replace=True,     # sample with replacement
                    n_samples=xdf_m.shape[0],    # to match majority class
                    random_state=123) # reproducible results

    totaldf = pd.concat([xdf_o,xdf_m])
    y = totaldf['class'].to_numpy() 
    X = totaldf.drop('class', axis=1).to_numpy()

    return X,y

def print_rocauc_curve(y_test, y_pred):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def evaluate_model(model, X_train, y_train, X_test, y_test):
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print("Acurácia Treino:",metrics.accuracy_score(y_train, y_pred))
    y_pred = clf.predict(X_test)
    print("Acurácia Teste:",metrics.accuracy_score(y_test, y_pred))
    y_proba = clf.predict_proba(X_test)
    print("AUC score:", metrics.roc_auc_score(y_test, y_proba[:,1]))
    print("F1 score:", metrics.f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print_rocauc_curve(y_test, y_pred)

    return y_pred

def cross_evaluate_model(model, X_train, y_train, X_test, y_test):
    clf = model.fit(X_train, y_train)
    y_pred = cross_val_predict(clf,X_train, y_train)
    print("Acurácia Treino:",metrics.accuracy_score(y_train, y_pred))
    y_pred = cross_val_predict(clf,X_test, y_test)
    print("Acurácia Teste:",metrics.accuracy_score(y_test, y_pred))
    y_proba = cross_val_predict(clf,X_test, y_test, method='predict_proba')
    print("AUC score:", metrics.roc_auc_score(y_test, y_proba[:,1]))
    print("F1 score:", metrics.f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print_rocauc_curve(y_test, y_pred)

    return y_pred


def printPCA(X, y):
    pca = PCA(2)  # project from 64 to 2 dimensions
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

def printBoxPlot(X):
    df = pd.DataFrame(data=X)
    display(df.describe())
    df.boxplot(figsize=(15,7))
    plt.show()

def beep(times, freq):
    duration = 0.5  # seconds

    for i in range(times):
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        time.sleep(0.5)



def plot_decision_boundaries(X, y, model_class, **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting 
    the model as we need to find the predicted value for every point in 
    scatter plot.
    Arguments:
            X: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model_class: A Scikit-learn ML estimator class 
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            LogisticRegression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator
    
    Typical code example:
            plt.figure()
            plt.title("KNN decision boundary with neighbros: 5",fontsize=16)
            plot_decision_boundaries(X_train,y_train,KNeighborsClassifier,n_neighbors=5)
            plt.show()
    """
    try:
        X = np.array(X)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")
    # Reduces to the first two columns of data
    reduced_data = X[:, :2]
    # Instantiate the model object
    model = model_class(**model_params)
    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].    

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Feature-1",fontsize=15)
    plt.ylabel("Feature-2",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return plt


