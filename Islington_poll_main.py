import os
from London_poll_preprocess import *
from London_poll_models import *


def create_and_get_folder_diffpoll (model_folder):
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


def preprocess_diffpoll (files_folders, pollutant_names, site):
    '''preprocessing of dataset, which contains datetimes and corresponding pollutant values
    Parameters:
    files_folders: a dictionary of files and folder names as keys and their directories as values,
    pollutant_names: a list of names (strings), site: string'''

    values_filename_diffpoll = files_folders['values_filename']
    dates_filename_diffpoll = files_folders['dates_filename']
    data_folder = files_folders['data_folder']

    ''' dfs is a dictionary with key: pollutant name and value: dataframe for each pollutant'''
    dfs = dict()
    for name in pollutant_names:
        data_filename = data_folder + site + '/' + name + '.csv'
        dfs[name] = read_data(data_filename)

    ''' counting the NaN'''
    for name in pollutant_names:
        print('Number of valid values for', name, (dfs[name]['Value'] > 0).sum())

    ''' temp_dates is a vector with all dates of the dataset'''
    temp_dates = dfs['NO']['ReadingDateTime'].values

    ''' temp_values is a dictionary with key:pollutant name and value: values for pollutant'''
    temp_values = dict()
    for name in pollutant_names:
        temp_values[name] = dfs[name]['Value'].values
        print('the shape of the array containing the values of', name, 'in Islington is', temp_values[name].shape)

    '''converting dates to datetime object'''
    dates = dates_to_dtime(temp_dates)

    '''getting hourly data for  each pollutant (every hour starting from 1/1/18 00:00)
        hourly_values is a dictionary with key:pollutant name and value: hourly values for pollutant'''
    hourly_values = dict()
    for name in pollutant_names:
        hourly_dates, hourly_values[name] = values_per_hour(dates, temp_values[name])

    '''geting data every 6 hours
        values_per_6h is a dictionary with key: pollutant name and value: pollutant values per 6 hours'''
    values_per_6h = dict()
    for name in pollutant_names:
        values_per_6h[name], final_dates = val_per_timeperiod(hourly_values[name], hourly_dates, num_hours = 6)
        temp = np.where(np.isnan(values_per_6h[name]))
        print('the fraction of nan values in', name, 'dataset for', site, 'is:',
              values_per_6h[name][temp].shape[0] / values_per_6h[name].shape[0])

    '''handling nan values
       final_values is a dictionary with key:site name and value:final pollutant values'''
    final_values = dict()
    for name in pollutant_names:
        final_values[name] = handle_nan(values_per_6h[name])
        temp = np.where(np.isnan(final_values[name]))
        print('the fraction of nan values in', name, 'dataset for', site, ',after handling Nans, are:',
              final_values[name][temp].shape[0] / final_values[name].shape[0])

    '''saving final values and dates in pickle files'''
    pickle.dump(final_values, open(values_filename_diffpoll, 'wb'))
    pickle.dump(final_dates, open(dates_filename_diffpoll, 'wb'))


def visualization_diffpoll (files_folders, pollutant_names, days_of_week, site):
    '''visualization of the data
    Parameters:
   files_folders: a dictionary of file and folder names as keys and their directories as values,
    pollutant_names: a list of names (strings), days_of_week: a list of weekday names, site: string'''

    values_filename = files_folders['values_filename']
    dates_filename = files_folders['dates_filename']
    img_folder = files_folders['img_folder']
    resultsfolder = files_folders['results_folder']

    final_values = pickle.load(open(values_filename, 'rb'))
    final_dates = pickle.load(open(dates_filename, 'rb'))

    '''the moving averaged values 
    mov_avg_values is a dictionary with key:pollutant name and value:moving averaged values for pollutant'''
    mov_avg_values = dict()
    for name in pollutant_names:
        mov_avg_values[name] = moving_average(final_values[name], 10)

    '''converting datetimes to weekdays'''
    days_arr = date_to_weekdays(final_dates)

    '''getting mean and standard deviation of values per weekday
    mean_values, std_values are 2 dictionaries with key:pollutant name and value:mean values/ standard deviation of
    pollutant values for every weekday of the year '''
    mean_values = dict()
    std_values = dict()
    for name in pollutant_names:
        mean_values[name], std_values[name] = val_per_weekday(days_arr, final_values[name], days_of_week)
        print('the mean values of', name, 'for each weekday of the year in Islington are', mean_values[name])

    '''plotting histograms for all pollutants'''
    poll_hist_diffpoll = histog_diffpoll (final_values, site, pollutant_names, img_folder)

    '''plotting pollutant moving averaged values for all pollutants'''
    poll_plot = plot_poll(img_folder, final_dates, mov_avg_values,  pollutant_names, site)

    '''getting correlation matrix for all datasets and the corresponding heatmap'''
    p_corr_diffpoll= pears_corr_diffpoll(final_values, site, pollutant_names, resultsfolder, img_folder)

    '''getting errorbar for each pollutant'''
    for name in pollutant_names:
        bar_diffpoll = barplot_diffpoll(mean_values[name], std_values[name], days_of_week, name, site, img_folder)


def model_results_diffpoll (files_folders2, pollutant_names, site, num_of_features):
    '''Saves the results for each site, model and number of features in csv files,
    saves in a pickle file the best model and the final conclusions in a txt file
    Parameters:
    files_folders: a dictionary of file and folder names as keys and their directories as values,
    pollutant_names: a list of names (strings), site: string, num_of_features: number of features of the datasets'''

    resultsfolder = files_folders2['results_folder']
    picklesfolder = files_folders2['pickle_folder']
    values_filename_diffpoll = files_folders2['values_filename']

    poll_values = pickle.load(open(values_filename_diffpoll, 'rb'))
    model_names = ['svm', 'rf', 'lasso', 'knn']

    '''getting results for each pollutant, model and number of features'''
    for name in pollutant_names:
        r2score_max = -10000
        for num_feat in range(1, num_of_features + 1):
            X, y = model_dataset(poll_values[name], num_feat)
            for model_name in model_names:
                ytest, ypred, index_test, index_train, mymodel = poll_models(model_name, X, y, picklesfolder, site,
                                                                             name)
                r2score = r2_score(ytest, ypred)
                print('r2score:', r2score)

                ypred = ypred.reshape(-1, 1)
                ytest = ytest.reshape(-1, 1)
                index = index_test.reshape(-1, 1)
                results = np.concatenate((index, ypred, ytest), axis=1)

                with open(resultsfolder + name + '_' + site + '_' + model_name + '_' + str(num_feat) + 'feat.csv',
                          'w') as wrt:
                    title = np.array((str('index'), str('predicted_value'), str('actual_value')))
                    writer = csv.writer(wrt, delimiter=',')
                    writer.writerow(title)
                    for line in results:
                        writer.writerow(line)

                if r2score > r2score_max:
                    best_num_features = num_feat
                    best_model = model_name
                    r2score_max = r2score
                    filename = picklesfolder + name + '_' + site + '_best_model.p'
                    pickle.dump(mymodel, open(filename, 'wb'))
        with open(resultsfolder + name + '_' + site + '_final_results.txt', 'w') as wrt:
            wrt.write('the best model for predicting ' + name + ' in ' + site + ' is: ' + str(best_model) +
                      ', the corresponding number of features is: ' + str(best_num_features) + '  and r2 score is: '
                      + str(r2score_max))


def main_diffpoll(data_folder, model_folder, site, num_features):

    pickle_folder, img_folder, results_folder = create_and_get_folder_diffpoll(model_folder)
    pollutant_names = ['NO2', 'NO', 'NOx', 'PM10']
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    values_filename_diffpoll = pickle_folder + site + '_' + '_diff_pollutant_final_values.p'
    dates_filename_diffpoll = pickle_folder + site + '_' + '_diff_pollutant_final_dates.p'

    files_and_folders_diffpoll = {}
    files_and_folders_diffpoll['data_folder'] = data_folder
    files_and_folders_diffpoll['pickle_folder'] = pickle_folder
    files_and_folders_diffpoll['img_folder'] = img_folder
    files_and_folders_diffpoll['results_folder'] = results_folder
    files_and_folders_diffpoll['values_filename'] = values_filename_diffpoll
    files_and_folders_diffpoll['dates_filename'] = dates_filename_diffpoll

    preprocess_diffpoll(files_and_folders_diffpoll, pollutant_names, site)
    visualization_diffpoll(files_and_folders_diffpoll, pollutant_names, weekdays, site)
    model_results_diffpoll(files_and_folders_diffpoll, pollutant_names, site, num_features)
    r2score, mse = eval_metrics_diffpoll(files_and_folders_diffpoll, site, pollutant_names, num_features)
    highest_metric_per_model_diffpoll(files_and_folders_diffpoll, site, pollutant_names, num_features)

    
main_diffpoll(data_folder = '/Users/marialiatsikou/Documents/coding practice datasets/LONDON AIR/',
      model_folder = '/Users/marialiatsikou/Documents/Code/practice/Londonair/',  site = 'Islington', num_features = 16)
