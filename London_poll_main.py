from London_poll_preprocess import *
from London_poll_models import *
import os


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


def preprocess (files_folders, site_names, pollutant):
    '''preprocessing of dataset, which contains datetimes and corresponding pollutant values
    Parameters:
    files_folders: a dictionary of file and folder names as keys and their directories as values,
    site_names: a list of names (strings), pollutant: string'''

    values_filename_diffsites = files_folders['values_filename']
    dates_filename_diffsites = files_folders['dates_filename']
    data_folder = files_folders ['data_folder']

    '''dfs is a dictionary with key:site name and value: dataframe of dates and pollutant values'''
    dfs = dict()
    for name in site_names:
        data_filename = data_folder + pollutant + '/' + pollutant + '_' + name + '.csv'
        dfs[name] = read_data(data_filename)
        #print(dfs[name].describe())

    ''' counting the NaN'''
    for name in site_names:
        print('Number of valid values of', pollutant, 'for', name, (dfs[name]['Value'] >0).sum())

    ''' temp_dates is a vector with all dates of the dataset'''
    temp_dates = dfs['Bexley']['ReadingDateTime'].values

    ''' temp_values is a dictionary with key:site name and value: pollutant values'''
    temp_values = dict()
    for name in site_names:
        temp_values[name] = dfs[name]['Value'].values

    '''converting dates to datetime object'''
    dates = dates_to_dtime(temp_dates)

    '''getting hourly data for each site (every hour starting from 1/1/18 00:00)
    hourly_values is a dictionary with key:site name and value: hourly pollutant values'''
    hourly_values = dict()
    for name in site_names:
        hourly_dates, hourly_values[name] = values_per_hour(dates, temp_values[name])

    '''geting data every 6 hours
    values_per_6h is a dictionary with key:site name and value: pollutant values per 6 hours'''
    values_per_6h = dict()
    for name in site_names:
        values_per_6h[name], final_dates = val_per_timeperiod(hourly_values[name], hourly_dates, num_hours = 6)
        temp = np.where(np.isnan(values_per_6h[name]))
        print('the fraction of nan values in pollutant dataset of', name, 'is:',
                values_per_6h[name][temp].shape[0] / values_per_6h[name].shape[0])

    '''handling nan values
    final_values is a dictionary with key:site name and value:final pollutant values'''
    final_values = dict()
    for name in site_names:
        final_values[name] = handle_nan(values_per_6h[name])
        temp = np.where(np.isnan(final_values[name]))
        print('the fraction of nan values in pollutant dataset, after handling Nans, of', name, 'are:',
              final_values[name][temp].shape[0]/final_values[name].shape[0])

    '''saving final values and dates in pickle files'''
    pickle.dump(final_values, open(values_filename_diffsites, 'wb'))
    pickle.dump(final_dates, open(dates_filename_diffsites, 'wb'))


def visualization (files_folders, site_names, days_of_week, pollutant):
    '''visualization of the data
    Parameters:
   files_folders: a dictionary of file and folder names as keys and their directories as values,
    site_names: a list of names (strings), days_of_week: a list of weekday names, pollutant: string'''

    values_filename = files_folders['values_filename']
    dates_filename = files_folders['dates_filename']
    img_folder = files_folders['img_folder']
    resultsfolder = files_folders['results_folder']

    final_values = pickle.load(open(values_filename, 'rb'))
    final_dates = pickle.load(open(dates_filename, 'rb'))

    '''moving averaged values
     mov_avg_values is a dictionary with key:site name and value:moving averaged pollutant values '''
    mov_avg_values = dict()
    for name in site_names:
        mov_avg_values[name] = moving_average(final_values[name], 10)

    '''convert datetimes to weekdays'''
    days_arr = date_to_weekdays(final_dates)

    '''getting mean and standard deviation of values per weekday
    mean_values, std_values are 2 dictionaries with key:site name and value:mean values/ standard deviation of values 
    for pollutant for every weekday of the year '''
    mean_values = dict()
    std_values = dict()
    for name in site_names:
        mean_values[name], std_values[name] = val_per_weekday(days_arr, final_values[name], days_of_week)
        #print('the mean values of NO2 for each weekday of the year in', name, 'are', mean_values[name])

    '''plotting histograms for all sites'''
    #poll_hist_diffsites = histog_diffsites (final_values, site_names, pollutant, img_folder)

    '''getting correlation matrix for all datasets and the corresponding heatmap'''
    p_corr_diffsites = pears_corr_diffsites(final_values, site_names, pollutant, resultsfolder, img_folder)

    '''getting errorbar for each site'''
    for name in site_names:
        bar_diffsites = barplot_diffsites(mean_values[name], std_values[name], days_of_week, name, pollutant, img_folder)


def model_results (files_folders1, site_names, pollutant, num_of_features):
    '''Saves the results for each site, model and number of features in csv files,
    saves in a pickle file the best model for each site and the final conclusions in a txt file
    Parameters:
    files_folders: a dictionary of file and folder names as keys and their directories as values,
    site_names: a list of names (strings), pollutant: string, num_of_features: number of features of the datasets'''

    resultsfolder = files_folders1['results_folder']
    picklesfolder = files_folders1['pickle_folder']
    values_filename_diffsites = files_folders1['values_filename']

    poll_values = pickle.load(open(values_filename_diffsites, 'rb'))
    model_names = ['svm', 'rf', 'lasso', 'knn']

    '''getting results for each site, model and number of features'''
    for name in site_names:
        r2score_max = -2
        for num_feat in range(1,num_of_features+1):
            X, y = model_dataset(poll_values[name], num_feat)
            for model_name in model_names:
                ytest, ypred, index_test, index_train, mymodel = poll_models (model_name, X, y, picklesfolder, name, pollutant)
                r2score = r2_score(ytest, ypred)
                print('r2score:',r2score)

                ypred = ypred.reshape(-1, 1)
                ytest = ytest.reshape(-1, 1)
                index = index_test.reshape(-1, 1)
                results = np.concatenate((index, ypred, ytest), axis=1)

                with open(resultsfolder + pollutant +'_' +name + '_'  + model_name +'_' +str(num_feat) +'feat.csv', 'w') as wrt:
                    title = np.array((str('index'), str('predicted_value'), str('actual_value')))
                    writer = csv.writer(wrt, delimiter=',')
                    writer.writerow(title)
                    for line in results:
                        writer.writerow(line)

                if r2score > r2score_max:
                    best_num_features = num_feat
                    best_model = model_name
                    r2score_max = r2score
                    filename = picklesfolder + pollutant +'_' + name + '_best_model.p'
                    pickle.dump(mymodel, open(filename, 'wb'))
        with open(resultsfolder + pollutant +'_' + name +'_final_results.txt', 'w') as wrt:
            wrt.write('the best model for predicting ' + pollutant +' in ' + name + ' is: ' +str(best_model)+
            ', the corresponding number of features is: ' +str (best_num_features) + '  and r2 score is: '
            + str(r2score_max) )


def main_diffsites (data_folder, model_folder, pollutant, num_features):

    pickle_folder, img_folder, results_folder = create_and_get_folder(model_folder)
    site_names = ['Bexley', 'Brent', 'Camden', 'City', 'Croydon', 'Ealing', 'Greenwich', 'Islington',
                  'Kensington', 'Westminster']
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    values_filename_diffsites = pickle_folder + pollutant + '_' + '_diff_sites_final_values.p'
    dates_filename_diffsites = pickle_folder + pollutant + '_' + '_diff_sites_final_dates.p'

    files_and_folders_diffsites = {}
    files_and_folders_diffsites['data_folder'] = data_folder
    files_and_folders_diffsites['pickle_folder'] = pickle_folder
    files_and_folders_diffsites['img_folder'] = img_folder
    files_and_folders_diffsites['results_folder'] = results_folder
    files_and_folders_diffsites['values_filename'] = values_filename_diffsites
    files_and_folders_diffsites['dates_filename'] = dates_filename_diffsites

    preprocess(files_and_folders_diffsites, site_names, pollutant)
    visualization(files_and_folders_diffsites, site_names, weekdays, pollutant)
    model_results(files_and_folders_diffsites, site_names, pollutant, num_features)
    r2score, mse = eval_metrics_diffsites(files_and_folders_diffsites, pollutant, site_names, num_features)
    highest_metric_per_model_diffsites(files_and_folders_diffsites, site_names, num_features, pollutant)



main_diffsites( data_folder = '/Users/marialiatsikou/Documents/coding practice datasets/LONDON AIR/',
      model_folder = '/Users/marialiatsikou/Documents/Code/practice/Londonair/',  pollutant='NO2', num_features = 16)



