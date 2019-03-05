import numpy as np
import pandas as pd
from sklearn import metrics
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
import csv


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


def traintestSplit (X, y, pct1, pct2):
    '''Returns train, dev and test set of a dataset and the corresponding indices
    Parameters:
    X: a rank 2 array, y: vector of floats, pct1, pct2: float numbers
    Outputs: Xtrain, Xdev, Xtest: rank 2 arrays of floats, 
    ytrain, ydev, ytest: vectors of floats, index_train, index_test: vectors of
      integers'''
    
    index = np.arange(X.shape[0])
    thresh1 = pct1 * len(X)
    thresh2 = (pct1 + pct2) * len(X)
    Xtrain = X[0:int(thresh1),:]
    ytrain = y[0:int(thresh1)]
    index_train = index[0:int(thresh1)]
    Xdev = X[int(thresh1):int(thresh2),:]
    ydev = y[int(thresh1):int(thresh2)]
    index_dev = index[int(thresh1):int(thresh2)]
    Xtest = X[int(thresh2):,:]
    ytest= y[int(thresh2):]
    index_test = index[int(thresh2):]
    return Xtrain, Xdev, Xtest, ytrain, ydev, ytest, index_train, index_dev, index_test



def norm_attrib(Xtrain, Xdev, Xtest, ytrain, ydev, ytest):
    '''Returns normalized features and targets with zero mean and unit variance
    Parameters:
    Xtrain, Xdev, Xtest: rank 2 arrays of floats, ytrain, ydev, ytest: vectors
    of floats
    Outputs: X_train_std, X_dev_std, X_test_std: rank 2 arrays of floats, 
    y_train_std, y_dev_std, y_test_std: vectors of floats'''

    X_scaler = StandardScaler().fit(Xtrain)
    ytrain = np.reshape(ytrain, (-1,1))
    y_scaler =  StandardScaler().fit(ytrain)
    X_train_std = X_scaler.transform(Xtrain)
    X_dev_std = X_scaler.transform(Xdev)
    X_test_std = X_scaler.transform(Xtest)
    y_train_std = y_scaler.transform(ytrain)
    ydev =  np.reshape(ydev, (-1,1))
    y_dev_std = y_scaler.transform(ydev)
    ytest =  np.reshape(ytest, (-1,1))
    y_test_std = y_scaler.transform(ytest)

    return X_train_std, X_dev_std, X_test_std, y_train_std, y_dev_std, y_test_std


def nn_model(pickle_folder, pollutant, site_name, X, y, a, neurons, mini_batch, num_epochs):
    '''Returns a Regression Neural Network, actual and predicted values, indices and saves the model in a pickle file
    Parameters:
    pickle_folder: a directory path (string), pollutant, sitename: strings
    X: rank 2 array of the features' values,
    Y: vector of the target values, a: learning rate (list of floats), neurons:
    number of neurons in each layer (list of integers),
    mini_batch: size of mini batch (list of integers), num_epochs :number of 
    epochs for  the model (int)
    Outputs: a pickle file containing the model, 2 vectors of predicted and 
    actual values (floats), a vector of indices (integers), the model'''

    
    num_features = X.shape[1]
    X_train, X_dev, X_test, y_train, y_dev, y_test, index_train, index_dev, index_test = traintestSplit(X, y, 0.7, 0.15)
    X_train_norm, X_dev_norm, X_test_norm, y_train_norm, y_dev_norm, y_test_norm = norm_attrib (X_train, X_dev, X_test, y_train, y_dev, y_test)
    print('the shapes of X_train, X_dev, X_test are:', 
          X_train.shape, X_dev.shape, X_test.shape)
    print('the shapes of y_train, y_dev, y_test are:', 
          y_train.shape, y_dev.shape, y_test.shape)
    
    counter = 0
    r2score_dev_max = -2
    for neuron in neurons:
        for batch in mini_batch:
            for learning_rate in a:
                #inintializing NN
                regr_model = Sequential()
                regr_model.add(Dense(kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', input_dim = X_train.shape[1], 
                    units = neuron, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)))

                regr_model.add(Dense(kernel_initializer='glorot_uniform', 
                        bias_initializer='zeros', units = neuron, 
                        activation='relu', kernel_regularizer=regularizers.l2(0.01)))
                
                regr_model.add(Dense(kernel_initializer='glorot_uniform', 
                        bias_initializer='zeros', units = 1, activation='linear', 
                        kernel_regularizer=regularizers.l2(0.01)))
                
                #define optimizer
                optim = optimizers.Adam(lr=learning_rate, beta_1=0.9, 
                        beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                
                #compile model
                regr_model.compile(optimizer=optim, loss = 'mean_squared_error', 
                                   metrics=['mse'])
                
                #fit model
                regr_model.fit(X_train_norm, y_train_norm, batch_size=batch,
                               validation_data = (X_dev_norm, y_dev_norm),
                               epochs=num_epochs, verbose=1)
                
                #predict in dev set
                y_pred_dev = regr_model.predict(X_dev_norm)
                r2score = r2_score(y_dev_norm, y_pred_dev)
                
                if r2score > r2score_dev_max:
                    my_model = regr_model
                    r2score_dev_max = r2score 
                    best_learningrate = learning_rate
                    best_size_of_batch = batch
                    best_num_neurons = neuron
                    counter += 1
                    print(counter)

    y_pred = my_model.predict(X_test_norm)
    
    
    print('counter is:', counter)
    print('The best model has %s learning rate, %s batches and %s neurons'
          %(best_learningrate, best_size_of_batch, best_num_neurons))
    print('The best model has %s r2 score' %(r2score_dev_max))
    
    model_filename = pickle_folder + pollutant +'_' + site_name + '_NN_' + str(num_features) + 'feat.p'
    pickle.dump(my_model, open(model_filename, 'wb'))
    
    return y_test_norm, y_pred, index_test, my_model


def eval_metrics_diffsites (files_folders1, pollutant, site_names, num_feat):
    '''Returns r2 score and mse between predicted and target values and the corresponding plots
    Parameters:
    files_folders1: a dictionary of files and folder names as keys and their 
    directories as values, pollutant: string, site_names: list of names 
    (strings), num_feat : number of features of the datasets (float),
    Output: 2 dictionaries with key: site name, model name and number of 
    features and value: the corresponding metrics'''

    resultsfolder = files_folders1['results_folder']
    imgfolder = files_folders1['img_folder']

    features = np. arange(1, num_feat+1)
    r2score = {}
    mse = {}

    for name in site_names:
        r2_per_site = {}
        mse_per_site = {}
        temp_r2_list = []
        temp_mse_list = []
        
        for n_of_features in features:
            model_results = pd.read_csv (resultsfolder + pollutant +'_' +name + '_NN_' +str(n_of_features) +'feat.csv')
            y_actual = model_results['actual_value'].values
            y_pred =  model_results['predicted_value'].values
            temp1 = r2_score(y_actual, y_pred)
            temp2 = mean_squared_error(y_actual, y_pred)
            temp_r2_list.append(temp1)
            temp_mse_list.append(temp2)
            r2score ['r2_'+name+'_nn_'+str(n_of_features)+'feat'] = temp1
            mse ['mse_'+name+'_nn_'+str(n_of_features)+'feat'] = temp2
        r2_per_site[name] = temp_r2_list
        mse_per_site[name] = temp_mse_list
        
        r2_plot = plot_metrics(files_folders1, features, r2_per_site, name, pollutant, 'r2 score')
        mse_plot = plot_metrics(files_folders1, features, mse_per_site, name, pollutant,'mse')
        
    return r2score, mse


def plot_metrics (files_folders, num_features, metric_per_site, site, pollutant, metric_name):
    '''plot r2 score and mse for every model as a function of the number of features
    Parameters:
    files_folders2: a dictionary of files and folder names as keys and their 
    directories as values, num_features: a range of number of features (vector) , 
    metric_per_model: dictionary of metric values
    with key: model name and value: list of metric values for each number of 
    features,
    site, pollutant: strings, metric_name: kind of metric (string)'''

    imgfolder = files_folders['img_folder']

    plt.figure()
    
    plt.plot (num_features,metric_per_site[site], linewidth=0.5)
    plt.grid()
    plt.xlabel('number of features')
    plt.ylabel(metric_name)
    x_last = int(num_features[-1]+1)
    plt.xticks(np.arange(1,x_last,1))
    plt.title (metric_name + ' of Neural Network predicting the values of ' + pollutant + ' in ' + site)
    plt.savefig(imgfolder + 'NN_' + pollutant +'_' + site +'_' + metric_name +'_plot.png')
    plt.show()


    

    
