# This Python script was used to 

# Set up imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests, os, sys, zipfile, io
import itertools
import pickle

# pyspark imports
import pyspark
from pyspark.sql import Window
from pyspark.sql import functions as func

from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.sql import Window
import pyspark.sql.functions as F

from pyspark.ml.pipeline import Estimator, Model
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml import Pipeline

import shutil

# Initialize spark. Configurations set to prevent Java heap space errors.
spark = (pyspark.sql.SparkSession
         .builder
         .config('spark.executor.memory', '30g')
         .config('spark.driver.memory', '30g')
         .config('spark.driver.maxResultSize', '30g')
         .getOrCreate() 
        )

# Import the file and create the schema needed
# playtime_5_min.csv created by pulling steam_id (user id), appid (game id), and 
# playtime_forever (playtime since inception) from the MySQL databse table. See
# Query-Playtime.sql for the query, which pulled all entries with more than 5 min
# of playtime.

# NOTE: the playtime_5_min.csv in the Github only contains 20,000 rows of example data, not the full set the
# model was actually trained on due to file size limits.
filename = 'playtime_5_min.csv'
playtime = spark.read.csv(filename,
                        sep='\t',
                        inferSchema=True,
                        header=True)

for column in ['steamid','appid','playtime_forever']:
    playtime_df = playtime.withColumn(column, playtime[column].cast('int'))
    
# Convert steamids to ints by subtracting the minimum steamid (76561197960265729) from each id, based on the
# minimum value of the steamid, as the ALS model in pyspark does not accept long values for ids.
playtime_df = playtime_df.withColumn('steamid_int', func.col('steamid') - 76561197960265729)

# Train Test Split
train_val, test = playtime_df.randomSplit([0.75, 0.25], seed=42)

def KFold(df, n_folds = 3, seed = 42):
    weights = np.ones(n_folds)
    fold_list = df.randomSplit(weights, seed = seed)
    
    data_tuples = []
    
    for fold in fold_list:
        train_test = fold.randomSplit([0.75, 0.25], seed=42)
        data_tuples.append(tuple(train_test))
    
    return data_tuples

# Create a custom class: in order to fit the transformer on the training data for each cross validation fold 
# using the z-score on the training set, then transform the validation set and test set on the same scale.
# The fit function creates a dataframe with the average and standard deviation per the training set; this is
# then used in the transform function to divide the playtime in minutes into standard deviations away from the
# training set average.
class AverageEstimator(Estimator, HasInputCol, HasOutputCol):
    def fit(self, X):
                '''
        Creates a Spark dataframe containing the average and standard deviation on a per-game basis for all games
        in the input dataframe X.

        Parameters: self and a dataframe X that contains steamid, steamid_int, appid, and playtime_forever.

        Outputs: a self object wih a Spark dataframe (self.table) containing the appid, average playtime for each
                 appid, and standard deviation of the playtime for each appid.

        '''

        grp_window = Window.partitionBy('appid')
        avg_func = func.mean('playtime_forever')
        stddev_func = func.stddev('playtime_forever')
        
        X = X.withColumn('avg', avg_func.over(grp_window))
        X = X.withColumn('std_dev', stddev_func.over(grp_window))
        
        self.table = X
        for col in ['steamid', 'steamid_int', 'playtime_forever']:
            self.table = self.table.drop(col)
        
        self.table = self.table.dropDuplicates(subset=['appid'])
        
        return self
    
    def transform(self, X):
        '''
        Transforms a given Spark dataframe containing playtime into a z-score based on a previously defined fit.

        Parameters: self (which contains a table per the previous fit that is used to transform a dataframe X) and
                    X, a dataframe to be transformed by subtracting the average from self.table and dividing the
                    difference by the standard deviation from self.table.

        Output: a transformed dataframe with z-scores in the playtime_scaled column.

        '''

        X = X.alias('X')
        self.table = self.table.alias('self')
        X2 = (X.join(self.table,on = X['appid'] == self.table['appid'], how = 'left')
               .select('X.*','self.avg', 'self.std_dev')
             )
        
        X2 = X2.withColumn('playtime_scaled', (X2['playtime_forever'] - X2['avg']) / X2['std_dev'])
        X2 = X2.drop('avg')
        X2 = X2.drop('std_dev')
        X2 = X2.filter(func.isnan('playtime_scaled') == False)
        
        return X2
    
def transform_fold(fold):
    '''
    Calls the customer AverageEstimator class above to fit on training data, then transform both the train and
    test data on the fit.

    Parameters: fold is a tuple containing 2 Spark dataframes as elements: the first is a dataframe of the
                training set and the second is the dataframe of the test set.

    Output: a transformed train set dataframe and test set dataframe.
    '''

    avg = AverageEstimator()
    train_fit = avg.fit(fold[0])
    train_transform = avg.transform(fold[0])
    test_transform = avg.transform(fold[1])
    return train_transform, test_transform

# Define the ALS model used for recommendations. coldStartStrategy set to true to allow RMSE to be measured by 
# ignoring N/A values due to users/items not existing in the training set within each fold. implicitPrefs set to 
# true to run the model using a proxy for ratings, rather than explicit ratings. nonnegative set to False to allow
# negative model predictions, as standard deviations can be negative.

# The rank will be modified as part of the gridsearch and cross validation.
als = ALS(userCol='steamid_int',
          itemCol='appid',
          ratingCol='playtime_scaled',
          coldStartStrategy='drop',
          implicitPrefs = True,
          nonnegative = False,
          seed=42)

# Parameter grid used for custom gridsearch to determine the best model. The RMSE for the rank 30 model was the
# best, so it was used for Final_Model.py.
rank_list = [8, 10, 12, 16, 30, 40, 50]

# Evaluator for the model; RMSE calculated on standard deviations of playtime hours.
rmse_eval = RegressionEvaluator(labelCol='playtime_scaled',
                                predictionCol='prediction', 
                                metricName='rmse')

def cross_val(df, model, n_folds = 3, evaluator = rmse_eval):
    '''
    Creates cross folds and trains the model on each cross fold. Custom cross_validation defined rather than using
    PySpark's CrossValidator due to being unable to break out the training and test data on each fold to accomodate
    the customized fit-transform noted above.

    Parameters: df is the spark dataframe containing the training and validation data. n_folds is the number of
                folds to perform for cross validation. evaluator is the evaluator defined for the model (RMSE)

    Output: A tuple where the first element of contains the parameters for
            each model and the second element is the average RMSE for the folds.

    '''
    fold_list = KFold(df, n_folds)
    
    rmse_list = []
    
    for fold in fold_list:
        train, test = transform_fold(fold)
        train = train.na.drop()
        
        als_model = model.fit(train)
        
        als_pred = als_model.transform(test)
        
        rmse_list.append(evaluator.evaluate(als_pred))

    average_rmse = np.mean(rmse_list)
    
    return (model.extractParamMap(), average_rmse)

def grid_search(df, rank_list, n_folds = 3, evaluator = rmse_eval):
    '''
    Custom grid-search function to accomodate cross-validator.

    Parameters: df is the spark dataframe containing the training and validation data. n_folds is the number of
                folds to perform for cross validation. rank_list is the list of ranks to try for the ALS models.
                evaluator is the evaluator defined for the model (RMSE)

    Output: A list of tuples equal to the output of the cross_validator for each rank used.

    '''
    model_list = []
    result_list = []
    
    for rank in rank_list:
        als = ALS(userCol='steamid_int',
                  itemCol='appid',
                  ratingCol='playtime_scaled',
                  coldStartStrategy='drop',
                  implicitPrefs = True,
                  nonnegative = False,
                  seed=42)
        als.setRank(rank)
        model_list.append(als)
    
    for model in model_list:
        result_list.append(cross_val(df, model, n_folds, evaluator))
        
    return result_list

result = grid_search(train_val, rank_list)

print(result)

# Preserve the ranks and RMSE for the grid search above.
with open('rank_grid.pkl', 'wb') as f:
    pickle.dump(test, f)

