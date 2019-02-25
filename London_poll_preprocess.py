import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import calendar
import csv


def read_data (filename):
    '''Returns a dataframe for each site/ pollutant'''
    return pd.read_csv(filename)


def dates_to_dtime (values):
    '''Returns a vector of datetimes
    Parameters: a vector of strings
    Outputs: a vector of datetimes'''

    datetime_object = []
    for i in range(values.shape[0]):
        datetime_object.append(datetime.strptime(values[i], '%d/%m/%Y %H:%M'))
    dates = np.asarray(datetime_object)

    return dates


def values_per_hour(dtime, values):
    '''Returns 2 new vectors, one contains the datetimes every hour and the other the corresponding means of values
     Parameters:
     dtime: a vector of datetime,   values: a vector of floats
     Outputs: datetimes every hour, corresponding mean values'''

    min_datetime = dtime.min()
    min_datetime = min_datetime.replace(second=0,minute=0)  #round down to hour
    max_datetime = min_datetime + timedelta(hours=1)

    datetime_new = []
    values_new = []
    while  min_datetime<dtime.max():
        datetime_new.append(min_datetime)
        '''temp returns an empty array if there are no values in the datetime dataset satisfying the 3 conditons'''
        temp = values[(dtime>=min_datetime) & (dtime<max_datetime) & (values>0)]
        if len(temp) ==0:
            values_new.append(np.nan)
        else:
            mean_val = np.mean(temp)
            values_new.append(mean_val)
        min_datetime = max_datetime
        max_datetime = min_datetime + timedelta(hours=1)
    datetime_new = np.asarray(datetime_new)
    values_new = np.asarray(values_new)
    return datetime_new, values_new


def val_per_timeperiod (tseries, hourly_dates, num_hours):
    '''Returns 2 new vectors, one contains the datetimes every time period and the other the corresponding means of values
    Parameters:
    tseries: a vector of floats, hourly_dates: a vector of datetime, num_hours: float number
    Outputs: 2 vectors of datetimes and corresponding mean values '''

    tseries_per_period = []
    dates_per_period =[]
    temp_val = np.mean(tseries[0:num_hours])
    if np.isnan(temp_val):
        tseries_per_period.append(np.nan)
    else:
        tseries_per_period.append(temp_val)
    dates_per_period.append(hourly_dates[0])

    for i in range(1,(int(tseries.shape[0]/num_hours))):
        temp = i*num_hours
        temp_val = np.mean(tseries[(temp):(temp+num_hours)])
        if np.isnan(temp_val):
            tseries_per_period.append(np.nan)
        else:
            tseries_per_period.append(temp_val)
        dates_per_period.append(hourly_dates[i*num_hours])

    tseries_per_period = np.asarray(tseries_per_period)
    dates_per_period = np.asarray(dates_per_period)

    return tseries_per_period, dates_per_period



def handle_nan (values):
    '''Returns a vector of values after interpolating the nan values
    Parameters:
    values: a vector of floats
    Output: a vector of floats'''

    if np.isnan(values).sum() > 0:
        temp = values[np.isfinite(values)]
        count = 0  # to find y
        if np.isnan(values[0]):
            values[0] = temp[0]
            count += 1
        if np.isnan(values[-1]):
            values[-1] = temp[-1]
        for i in range(len(values)):
            if np.isnan(values[i]):
                j = i+1
                count1 = 2  #the denominator of grad
                while np.isnan(values[j]):
                    count1 +=1
                    j +=1
                y = temp[i-count]
                count += 1
                grad = (y-values[i-1]) / count1
                values[i] = values[i-1] + grad*1

    return values


def moving_average(arr, history):
    '''Returns the moving average of a given vector
    Parameters:
    arr: a vector of floats, history: the window size
    Output: the vector of moving averaged values with history window'''

    arr_new = np.zeros(arr.shape)
    arr_new[0] = arr[0]
    for s in range(1,history):
        arr_new[s] = np.mean(arr[0:s+1])
    for s in range(history , arr_new.shape[0]):
        arr_new[s] = np.mean(arr[(s-history+1):(s+1)])

    return arr_new


def date_to_weekdays (dates_arr):
    '''Returns a vector of the  corresponding weekdays of a given vector of datetimes
    Parameters:
    a vector of dates
    Output:a vector of weekdays'''

    weekdays = []
    for i in range(dates_arr.shape[0]):
        temp = calendar.day_name[dates_arr[i].weekday()]
        weekdays.append(temp)
    weekdays = np.asarray(weekdays)

    return weekdays


def val_per_weekday(days, values, days_of_week):
    '''Return 2 vectors of values (mean & std) per day of week
    Parameters:
    days, days_of_week: vectors of strings, values: vector of floats
    Outputs:2 vectors of values (mean & std) '''

    mean_val = []
    std_val = []
    for i in days_of_week:
        temp = values[np.where(days == i)]
        mean_temp = np.mean(temp)
        std_temp = np.std(temp)
        mean_val.append(mean_temp)
        std_val.append(std_temp)

    mean_val = np.asarray(mean_val)
    std_val = np.asarray(std_val)

    return mean_val, std_val


def histog_diffsites (values, sites, pollutant, imgfolder):
    '''Plot histogram of values for all sites
    Parameters:
    values: a dictionary [site:values], sites: a list of strings, pollutant: string,
    imgfolder: directory path (string)'''

    fig = plt.figure()
    hist_title = fig.suptitle('Histograms of ' +pollutant + ' values (ug/m3) in London in 2018', fontsize=10)
    position = 1
    for name in sites:
        plt.subplot(2, int(len(sites)/2), position)
        plt.hist(values[name],bins='auto', range = (np.nanmin(values[name]), np.nanmax(values[name])))
        plt.xlabel(pollutant + ' in ' + name, fontsize=6)
        plt.ylabel('frequency', fontsize=6)
        plt.xticks(np.arange(0,250,50),size=4)
        plt.yticks(size=4)
        position += 1
    plt.savefig(imgfolder + pollutant + '_diff_sites_histogram.png')
    #plt.show()


def histog_diffpoll (values, site, pollutants, imgfolder):
    '''Plot histogram of values for all pollutants
    Parameters:
    values: a dictionary [pollut_name:values], site: string, pollutants: a list of strings,
    imgfolder: directory path (string)'''

    fig = plt.figure()
    hist_title = fig.suptitle('Histograms of pollutant values (ug/m3) in '+ site +' in 2018', fontsize=10)
    position = 1
    for name in pollutants:
        plt.subplot(2, int(len(pollutants)/2), position)
        plt.hist(values[name],bins='auto', range = (np.nanmin(values[name]), np.nanmax(values[name])))
        plt.xlabel(name)
        plt.ylabel('frequency')
        #plt.xticks(np.arange(0,250,50),size=4)
        #plt.yticks(size=4)
        position += 1
    plt.savefig(imgfolder + site + '_diff_poll_histogram.png')
    #plt.show()


def plot_poll (imgfolder, dtime, mov_av_values, pollut_names, site):
    '''plot pollutant moving averaged values for various pollutants in a site
    Parameters:
    imgfolder: directory path (string) dtime: a vector of datetimes, mov_av_values: a dictionary [pollut_name:values],
    site: string'''

    plt.figure()
    for name in pollut_names:
        plt.plot(dtime, mov_av_values[name], label = 'moving averaged ' + name + ' values', linewidth=0.5)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('pollution values (ug/m3)')
    plt.legend()
    plt.title ('Pollution values in 2018 in ' +site)
    plt.savefig(imgfolder + site + 'poll_plot.png')
    #plt.show()


def pears_corr_diffsites(values, sites, pollutant, resultsfolder, imgfolder):
    '''Returns the Pearson correlation matrix between every pair of datasets
    Parameters:
    values: dictionary [site: values], sites: a list of strings, pollutant: string,
    resultsfolder, imgfolder: directory paths (strings)
    Output: a Pearson correlation coefficient matrix and a heatmap '''

    all_values = []
    for name in sites:
        all_values.append(values[name])
    all_values = np.asarray(all_values)
    all_val = all_values.reshape((len(sites),-1))

    corr = np.corrcoef(all_val)

    '''with open(resultsfold + pollutant +'_diff_sites_final_results.txt', 'w') as wrt:
        wrt.write('The correlation matrix for' + pollutant + 'values in Bexley, Brent, Camden, City_of_London, Croydon, '
                       'Ealing, Greenwich, Islington, Kensington_Chelsea, Westminster is \n')
        wrt.write('\n')
        wrt.write("%s" %corr)'''
    sites_arr = np.asarray(sites)
    sites_arr = np.reshape(sites_arr, (1,len(sites)))
    with open(resultsfolder + pollutant + '_diff_sites_cor_matrix.csv', 'w') as wrt:
        writer = csv.writer(wrt, delimiter=',')
        for line in sites_arr:
            writer.writerow(line)
        for line in corr:
            writer.writerow(line)

    '''create a heatmap for the values of the sites'''
    corr_hmap = sns.heatmap(corr, square=True, xticklabels=sites,
                            yticklabels=sites,
                            vmin=0, vmax=1, cbar_kws={"shrink": 1})

    corr_hmap.set_title('Heatmap of Pearson correlation between '+ pollutant+ ' values for the sites',fontsize = 14)
    corr_hmap.set_yticklabels(corr_hmap.get_yticklabels(), rotation = 45, fontsize = 8)
    corr_hmap.set_xticklabels(corr_hmap.get_xticklabels(), rotation = 45, fontsize = 8)
    sns.set(font_scale=2)

    figure = corr_hmap.get_figure()
    figure.savefig(imgfolder + pollutant + '_diff_sites_heatmap.png')
    plt.show()

    return corr


def pears_corr_diffpoll (values, site, pollutants,  resultsfolder, imgfolder):
    '''Returns the Pearson correlation matrix between every pair of datasets
    Parameters:
    values: dictionary [pollut_name:values],site: string, pollutants: a list of strings,
    resultsfolder, imgfolder: directory paths (strings)
    Output: a Pearson correlation coefficient matrix and a heatmap '''

    all_values = []
    for name in pollutants:
        all_values.append(values[name])
    all_values = np.asarray(all_values)
    all_val = all_values.reshape((len(pollutants),-1))

    corr = np.corrcoef(all_val)
    poll_arr = np.asarray(pollutants)
    poll_arr = np.reshape(poll_arr, (1,len(pollutants)))
    with open(resultsfolder + site + '_diff_poll_cor_matrix.csv', 'w') as wrt:
        writer = csv.writer(wrt, delimiter=',')
        for line in poll_arr:
            writer.writerow(line)
        for line in corr:
            writer.writerow(line)

    '''create a heatmap for the values of the sites'''
    corr_hmap = sns.heatmap(corr, square=True, xticklabels=pollutants,
                            yticklabels=pollutants,
                            vmin=0, vmax=1, cbar_kws={"shrink": 1})

    corr_hmap.set_title('Heatmap of Pearson correlation between pollutant values for ' + site,fontsize = 14)
    corr_hmap.set_yticklabels(corr_hmap.get_yticklabels(), fontsize = 8)
    corr_hmap.set_xticklabels(corr_hmap.get_xticklabels(), fontsize = 8)
    sns.set(font_scale=2)

    figure = corr_hmap.get_figure()
    figure.savefig(imgfolder + site + '_diff_poll_heatmap.png')
    #plt.show()

    return corr


def barplot_diffsites(y_data, error_data, days_of_week, sitename, pollutant, imgfolder):
    '''Returns the bar plot and error bar of values for all days
    Parameters:
    y_data, error_data: vectors of floats(mean, std), days_of_week: list of strings , sitename, pollutant:strings,
    imgfolder: directory path (string)
    Output: a bar plot & error bar'''

    days_in_num = [0, 1, 2, 3, 4, 5, 6] # to avoid sorting in strings
    _, ax = plt.subplots()
    '''Draw bars, position them in the center of the tick mark on the x-axis'''
    ax.bar(days_in_num, y_data, color='#539caf', align='center')
    '''Draw error bars to show standard deviation, set ls to 'none to remove line between points'''
    ax.errorbar(days_in_num, y_data, yerr=error_data, color='#297083', ls='none', lw=2, capthick=2)
    plt.xticks(np.arange(7), days_of_week, size=8)
    plt.yticks(size=8)
    plt.grid()
    plt.title('Bar plot and error bar by day of week for ' + pollutant + ' values in '+sitename+' in 2018', fontsize=12)
    plt.savefig(imgfolder +sitename+ pollutant+ '_bar.png')
    plt.show()


def barplot_diffpoll(y_data, error_data, days_of_week, pollutant, sitename, imgfolder):
    '''Returns the bar plot and error bar of values for all days
    Parameters:
    y_data, error_data: vectors of floats(mean, std), days_of_week: list of strings , pollutant, sitename:strings,
    imgfolder: directory path (string)
    Output: a bar plot & error bar'''

    days_in_num = [0, 1, 2, 3, 4, 5, 6] # to avoid sorting in strings
    _, ax = plt.subplots()
    '''Draw bars, position them in the center of the tick mark on the x-axis'''
    ax.bar(days_in_num, y_data, color='#539caf', align='center')
    '''Draw error bars to show standard deviation, set ls to 'none to remove line between points'''
    ax.errorbar(days_in_num, y_data, yerr=error_data, color='#297083', ls='none', lw=2, capthick=2)
    plt.xticks(np.arange(7), days_of_week, size=8)
    plt.yticks(size=8)
    plt.grid()
    plt.title('Bar plot and error bar by day of week for ' + pollutant + ' values in '+sitename+' in 2018', fontsize=12)

    plt.savefig(imgfolder +sitename+ pollutant+ '_bar.png')

    plt.show()

