# London_Air

This project aims at predicting the values of air pollution in various sites within Greater London. More specifically, the targets are the concentrations of various pollutants in different locations for the year 2018. The datasets are publicly  available from Londonair (https://www.londonair.org.uk/). The analysis is based on machine learning algorithms (SVMs, Random Forests, Lasso Regression, k-nearest neighbours and Feedforward Neural Network), leveraging historical data.
The analysis comprises 2 approaches:

∙ predicting the values of one pollutant (NO2) in 10 sites across London

∙ predicting the values of various pollutants (NO2, NOx, NO, PM10) in a single site (Islington).


The steps in both approaches are the following:

1.	Data preprocessing: the data are converted to consecutive six-hour bins and the missing values are linearly interpolated.
2.	Data visualization: the moving averaged time series for all pollutants in Islington and histograms for all sites and pollutants are plotted. Pearson correlation matrix in both cases is computed and the corresponding heatmap is plotted. Finally, bar plots and error bars of the mean values of pollution for every day of the week are provided.
3.	Construction of the dataset: The prediction models are using historical air pollution data as features. The number of features of each instance is the number of lagged variables used to predict the target value. We experiment with a range of such lagged variables (1-16).
4.	Train/ Test split and normalization: the dataset is split in training and test set using a percentage rate of 80/20. The test set contains the last values of our dataset. In the case of Neural Networks a development set is also used for evaluation with a split of 70/10/20. The values of the features of the training set are normalized to zero mean and unit variance. The same transformation is applied to the test set.
5.	Model training: each of the aforementioned algorithms is trained in both cases and the results are saved.
6.	Model evaluation: MSE and r2 score between predicted and target values are computed. The two metrics are plotted as a function of the number of features for each model.

