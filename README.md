# London_Air

This project aims at predicting the values of air pollution in various sites within Greater London. More specifically, the targets are the concentrations of various pollutants in different locations for the year 2018. The datasets are publicly  available from Londonair (https://www.londonair.org.uk/). The analysis is based on machine learning algorithms (SVMs, Random Forests, Lasso Regression, k-nearest neighbours and Feedforward Neural Network), leveraging historical data.
The analysis comprises 2 approaches:

∙ predicting the values of NO2 in 10 sites across London
∙ predicting the values of NO2, NOx, NO, PM10 in Islington.

The steps in both approaches are the following:
1.	data preprocessing: using an averaging period of 6 hours for the values and interpolating the missing values
2.	data visualization:  plot of the moving averaged timeseries for all pollutants in Islington, histograms for all sites and pollutants, Pearson correlation matrix in both cases and the corresponding heatmap, bar plots and error bars of the mean values for every day of the week
3.	Construction of the dataset: The prediction models are using historical air pollution data as features. The number of features of each instance is the number of lagged values used to predict the target value, with lag = 6 hours, and is a hyperparameter.
4.	Train/ Test split and normalization: splitting the dataset in training and test set using a rate and normalize the dataset to zero mean and unit variance (in the case of Neural Networks a development set is also used for hyperparameter tuning).
5.	Model training: training each of the aforementioned ML algorithms in both cases and keeping the results.
6.	Model evaluation: computing mse and r2 score between predicted and target values and the corresponding plots of these metrics as a function of the number of features for each model.

