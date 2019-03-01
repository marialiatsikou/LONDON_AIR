import numpy as np
import pandas as pd
from sklearn import metrics
import pickle
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


def model_dataset (tseries, num_features):
    '''Returns the dataset for our model, i.e. X,Y
    Parameters :
    tseries: vector of floats ,num_features: float number
    Output: a rank 2 array of features X, a vector of target values Y'''

    X = []
    Y = []
    for i in range(num_features, tseries.shape[0]):
        Y.append(tseries[i])
        for j in range (i-num_features, i):
            X.append(tseries[j])
    X = np.asarray(X)
    Y = np.asarray(Y)
    X = np.reshape(X, (-1, num_features))
    print('shapes of X,Y',X.shape, Y.shape)

    return X, Y


def traintestSplit (X,y,pct):
    '''Returns train and test set of a dataset
    Parameters:
    X: a rank 2 array, y: vector of floats, pct: float number
    Output: Xtrain, Xtest: rank 2 arrays of floats, ytrain, ytest: vectors of floats '''

    index = np.arange(X.shape[0])
    thresh = (1-pct)*len(X)
    Xtrain = X[0:int(thresh),:]
    ytrain = y[0:int(thresh)]
    index_train = index[0:int(thresh)]
    Xtest = X[int(thresh):,:]
    ytest= y[int(thresh):]
    index_test = index[int(thresh):]

    return Xtrain, Xtest, ytrain, ytest, index_train, index_test


def norm_attrib(Xtrain, Xtest, ytrain, ytest):
    '''Returns normalized features with zero mean and unit variance
    Parameters:
    Xtrain, Xtest: rank 2 arrays of floats, ytrain, ytest: vectors of floats
    Output: X_train_std, X_test_std: rank 2 arrays of floats, y_train_std, y_test_std: vectors of floats'''

    X_scaler = StandardScaler().fit(Xtrain)
    ytrain = np.reshape(ytrain, (-1,1))
    y_scaler =  StandardScaler().fit(ytrain)
    X_train_std = X_scaler.transform(Xtrain)
    X_test_std = X_scaler.transform(Xtest)
    y_train_std = y_scaler.transform(ytrain)
    ytest =  np.reshape(ytest, (-1,1))
    y_test_std = y_scaler.transform(ytest)

    return X_train_std, X_test_std, y_train_std, y_test_std


def poll_models (model_name, X, y, pickle_folder, site_name, pollutant):
    '''Returns a Regression prediction model, predicted and actual values, indices and saves the model in a pickle file
    Parameters:
    model_name: a string (defining which model to apply for regression), X: rank 2 array of the features' values,
    Y: vector of the target values, pickle_folder: a directory path (string), site_name, pollutant: strings
    Outputs: a pickle file containing the model, 2 vectors of predicted and actual values (floats),
    2 vectors of indices (integers), the model'''

    num_features = X.shape[1]
    X_train, X_test, y_train, y_test, index_train, index_test = traintestSplit(X, y, 0.2)
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = norm_attrib(X_train, X_test, y_train, y_test)

    if model_name == 'svm':
        regr_model = svm.SVR
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        gammas = [0.001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        parameters = {'C': Cs, 'gamma': gammas}
        my_model = GridSearchCV(regr_model(kernel='rbf', epsilon = 0.01), param_grid = parameters, cv=5)
    elif model_name == 'rf':
        regr_model = RandomForestRegressor
        num_estimators = [50, 10 ** 2, 2 * (10 ** 2), 3 * (10 ** 2), 4 * (10 ** 2), 5 * (10 ** 2)]
        parameters = {'n_estimators': num_estimators}
        my_model = GridSearchCV(regr_model(criterion='mse'), param_grid=parameters, cv=5)
    elif model_name == 'lasso':
        regr_model = Lasso
        a = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        parameters = {'alpha': a}
        my_model = GridSearchCV(regr_model(), param_grid = parameters, cv=5)
    elif model_name == 'knn':
        regr_model = KNeighborsRegressor
        num_neigh = [2, 5, 10, 20, 50]
        parameters = {'n_neighbors': num_neigh}
        my_model = GridSearchCV(regr_model(), param_grid=parameters, cv=5)
    y_train_norm = y_train_norm.ravel()
    my_model.fit(X_train_norm, y_train_norm)
    print(my_model.best_estimator_)
    y_pred = my_model.predict(X_test_norm)

    model_filename = pickle_folder + pollutant +'_' + site_name + '_'  + model_name + '_' + str(num_features) + 'feat.p'
    pickle.dump(my_model, open(model_filename, 'wb'))

    return y_test_norm, y_pred, index_test, index_train, my_model


def eval_metrics_diffsites (files_folders1, pollutant, site_names, num_feat):
    '''Returns r2 score and mse between predicted and target values and the corresponding plots
    Parameters:
    files_folders1: a dictionary of files and folder names as keys and their directories as values, pollutant: string,
    site_names: list of names (strings), num_feat : number of features of the datasets (float),
    Output: 2 dictionaries with key: site name, model name and number of features and value: the corresponding metrics'''

    resultsfolder = files_folders1['results_folder']
    imgfolder = files_folders1['img_folder']

    model_names = ['svm', 'rf', 'lasso', 'knn']
    features = np. arange(1, num_feat+1)
    r2score = {}
    mse = {}

    for name in site_names:
        r2_per_model = {}
        mse_per_model = {}

        for model_name in model_names:
            temp_r2_list = []
            temp_mse_list = []
            for n_of_features in features:
                model_results = pd.read_csv (resultsfolder + pollutant +'_' +name + '_'  + model_name +'_' +str(n_of_features) +'feat.csv')
                y_actual = model_results['actual_value'].values
                y_pred =  model_results['predicted_value'].values
                temp1 = r2_score(y_actual, y_pred)
                temp2 = mean_squared_error(y_actual, y_pred)
                temp_r2_list.append(temp1)
                temp_mse_list.append(temp2)
                r2score ['r2_'+name+'_'+model_name+'_'+str(n_of_features)+'feat'] = temp1
                mse ['mse_'+name+'_'+model_name+'_'+str(n_of_features)+'feat'] = temp2
            r2_per_model[model_name] = temp_r2_list
            mse_per_model[model_name] = temp_mse_list
        r2_plot = plot_metrics(files_folders1, features, r2_per_model, model_names, name, pollutant, 'r2 score')
        mse_plot = plot_metrics(files_folders1, features, mse_per_model, model_names, name, pollutant,'mse')

    return r2score, mse


def eval_metrics_diffpoll (files_folders2, site, pollutant_names, num_feat):
    '''Returns r2 score and mse between predicted and target values and the  corresponding plots
    Parameters:
    files_folders2: a dictionary of files and folder names as keys and their directories as values,
    pollutant_names: list of names (strings), site: string, num_feat: number of features of the datasets (float),
    Output: 2 dictionaries with key: pollutant name, model name and number of features and value: the corresponding metrics'''

    resultsfolder = files_folders2['results_folder']
    imgfolder = files_folders2['img_folder']

    model_names = ['svm', 'rf', 'lasso', 'knn']
    features = np. arange(1, num_feat+1)
    r2score = {}
    mse = {}

    for name in pollutant_names:
        r2_per_model = {}
        mse_per_model = {}

        for model_name in model_names:
            temp_r2_list = []
            temp_mse_list = []
            for n_of_features in features:
                model_results = pd.read_csv (resultsfolder + name +'_' +site + '_'  + model_name +'_' +str(n_of_features) +'feat.csv')
                y_actual = model_results['actual_value'].values
                y_pred =  model_results['predicted_value'].values
                temp1 = r2_score(y_actual, y_pred)
                temp2 = mean_squared_error(y_actual, y_pred)
                temp_r2_list.append(temp1)
                temp_mse_list.append(temp2)
                r2score ['r2_'+name+'_'+model_name+'_'+str(n_of_features)+'feat'] = temp1
                mse ['mse_'+name+'_'+model_name+'_'+str(n_of_features)+'feat'] = temp2
            r2_per_model[model_name] = temp_r2_list
            mse_per_model[model_name] = temp_mse_list
        r2_plot = plot_metrics(files_folders2, features, r2_per_model, model_names, site, name, 'r2 score')
        mse_plot = plot_metrics(files_folders2, features, mse_per_model, model_names, site, name,'mse')

    return r2score, mse


def plot_metrics (files_folders, num_features, metric_per_model, model_names, site, pollutant, metric_name):
    '''plot r2 score and mse for every model as a function of the number of features
    Parameters:
    files_folders2: a dictionary of files and folder names as keys and their directories as values,
    num_features: a range of number of features (vector) , metric_per_model: dictionary of metric values
    with key: model name and value: list of metric values for each number of features,
    site, pollutant: strings, metric_name: kind of metric (string)'''

    imgfolder = files_folders['img_folder']

    plt.figure()
    for model_name in model_names:
        plt.plot (num_features,metric_per_model[model_name], label = (model_name) , linewidth=0.5)
    plt.grid()
    plt.xlabel('number of features')
    plt.ylabel(metric_name)
    x_last = int(num_features[-1]+1)
    plt.xticks(np.arange(1,x_last,1))
    plt.legend()
    plt.title (metric_name + ' of models predicting the values of ' + pollutant + ' in ' + site)
    plt.savefig(imgfolder + pollutant +'_' + site +'_' + metric_name +'_plot.png')
    plt.show()

