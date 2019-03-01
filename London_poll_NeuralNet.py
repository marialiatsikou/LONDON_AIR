#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:43:29 2019

@author: marialiatsikou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import pickle
from sklearn.metrics import r2_score, mean_squared_error
import os
from London_poll_nn_model import *

def create_and_get_folder (model_folder):
    '''Returns 3 directory paths of folders
    Parameters:
    model_folder: a directory path of the main folder
    Outputs: directory paths (strings)'''

    pickle_folder = model_folder + 'pickles/'
    img_folder = model_folder + 'img/'
    results_folder = model_folder + 'results/'
    folders = [pickle_folder, img_folder, results_folder]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(pickle_folder)

    return pickle_folder, img_folder, results_folder


def model_results (files_folders1, site_names, pollutant, num_of_features):
    '''Saves the results for each site and number of features in csv files,
    saves in a pickle file the best model for each site and the final conclusions 
    in a txt file
    Parameters:
    files_folders: a dictionary of file and folder names as keys and their 
    directories as values, site_names: a list of names (strings), 
    pollutant: string, num_of_features: number of features of the datasets'''

    resultsfolder = files_folders1['results_folder']
    picklesfolder = files_folders1['pickle_folder']
    values_filename_diffsites = files_folders1['values_filename']

    poll_values = pickle.load(open(values_filename_diffsites, 'rb'))
    
    learning_rate = [0.0001, 0.001, 0.01]
    num_of_neurons = [10, 25, 50, 100, 200]
    mini_batch_size = [32, 64]

    '''getting results for each site, model and number of features'''
    for name in site_names:
        r2score_max = -2
        for num_feat in range(1,num_of_features+1):
            X, y = model_dataset(poll_values[name], num_feat)
            ytest, ypred, index_test, mymodel = nn_model (picklesfolder, pollutant, name,
                    X, y, learning_rate, num_of_neurons, mini_batch_size, num_epochs = 50)
            r2score = r2_score(ytest, ypred)
            print('r2score:',r2score)

            ypred = ypred.reshape(-1, 1)
            ytest = ytest.reshape(-1, 1)
            index = index_test.reshape(-1, 1)
            results = np.concatenate((index, ypred, ytest), axis=1)

            with open(resultsfolder + pollutant +'_' +name + '_NN_' +
                      str(num_feat) +'feat.csv', 'w') as wrt:
                title = np.array((str('index'), str('predicted_value'),str('actual_value')))
                writer = csv.writer(wrt, delimiter=',')
                writer.writerow(title)
                for line in results:
                    writer.writerow(line)

            if r2score > r2score_max:
                best_num_features = num_feat
                r2score_max = r2score
                filename = picklesfolder + pollutant +'_' + name + '_ NN_model.p'
                pickle.dump(mymodel, open(filename, 'wb'))
        with open(resultsfolder + pollutant +'_' + name +'_NN_final_results.txt', 'w') as wrt:
            wrt.write('the best Neural Network for predicting ' + pollutant +' in ' + 
                      name +' corresponds to: ' +str (best_num_features) + 
                      '  number of features and r2 score is: '+ str(r2score_max) )
            
            
def main_diffsites (data_folder, model_folder, pollutant, num_features):

    pickle_folder, img_folder, results_folder = create_and_get_folder(model_folder)
    site_names = ['Bexley', 'Brent', 'Camden', 'City', 'Croydon', 'Ealing', 
                  'Greenwich', 'Islington', 'Kensington', 'Westminster']

    values_filename_diffsites = pickle_folder + pollutant + '_' + '_diff_sites_final_values.p'
    dates_filename_diffsites = pickle_folder + pollutant + '_' + '_diff_sites_final_dates.p'

    files_and_folders_diffsites = {}
    files_and_folders_diffsites['data_folder'] = data_folder
    files_and_folders_diffsites['pickle_folder'] = pickle_folder
    files_and_folders_diffsites['img_folder'] = img_folder
    files_and_folders_diffsites['results_folder'] = results_folder
    files_and_folders_diffsites['values_filename'] = values_filename_diffsites
    files_and_folders_diffsites['dates_filename'] = dates_filename_diffsites
    
    model_results(files_and_folders_diffsites, site_names, pollutant, num_features)
    r2score, mse = eval_metrics_diffsites(files_and_folders_diffsites, pollutant, site_names, num_features)
    print(mse, r2score)


main_diffsites( data_folder = 
               '/Users/marialiatsikou/Documents/coding practice datasets/LONDON AIR/',
      model_folder = '/Users/marialiatsikou/Documents/Code/practice/Londonair/',  
      pollutant='NO2', num_features = 16)
