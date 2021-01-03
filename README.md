# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about a marketing campaigns on bank clients and whether they will subscribe to a fixed term deposit.
We seek to predict if the client will subscribe a term deposit (variable y) which is later identified as the target column for predictions

The best performing model was the prefitted soft VotingClassifier with an accuracy of 0.9166. It was obtained with autoML. The custom model LogisticRegression with HyperDrive reached an accuracy of 0.9088 

## Scikit-learn Pipeline
Azure Machine Learning HyperDrive will be used to automate hyperparameter tuning and run experiments in parallel to efficiently optimize hyperparameters.

We start coding the custom model in a training script which will be run by HyperDrive. The model chosen is a logistic regression from scikit-learn.

Logistic Regression is used when the target variable is categorical. In our case, the categorical response has only two 2 possible outcomes: subscribe to fixed deposit or not

The training script will create a dataset from csv file. The dataset is cleaned by converting categorical variable into dummy/indicator variables and applying one hot encoding for month and for days of the week. We then split the dataset into 25% for test and 75% for training. 

The training script will train the logistic regression using 2 hyperparameters:
* --C: Inverse of regularization strength. Regularization is applying a penalty to increasing the magnitude of parameter values in order to reduce overfitting. Smaller values cause stronger regularization. So we have chosen value between 0.05 and 0.1
* --max_iter : Maximum number of iterations to converge. We have chosen values less than 100.

The trained model is then saved in a file and its accuracy metrics is logged to Azure ML run using the Run object within the script.

With notebook using Hyperdrive we are going to find optimal hyperparameter for the logistic regression. We define a search space with two parameters - C and max_iter. C is a continuous hyperparameter specified as a distribution over a continuous range of values. Max_iter is a discrete hyperparameter specified as a choice among discrete values.

We have chosen random sampling as the parameter sampling method to use over the hyperparameter. In random sampling, hyperparameter values are randomly selected from the defined search space. The random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. 

The primary metric is accuracy. The goal of the hyperparameter tuning will be to MAXIMIZE the accuracy. Each training run is evaluated for the primary metric. The early termination policy uses the primary metric to identify low-performance runs. Bandit policy will be used as early termination policy. It is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. We have chosen a slack factor of 0.1 with an evaluation interval of 1.


## AutoML
We have run AutoML classification experiment using accuracy as the primary metric with experiment timeout minutes set to 30 minutes and 5 cross-validation folds. After 25 methods tested, the best model obtained is VotingEnsemble with an accuracy of 0.9166 and all hyperparameters tunned. We found hyperparameters to control the growth of Decision Trees (min_samples_leaf, and so on), as well as hyperparameters to control the ensemble training, such as the number of trees (n_estimators).

## Pipeline comparison
The following steps must be completed using Hyperdrive : 
*	Clean data
*	Define the parameter search space
*	Specify a primary metric to optimize
*	Specify early termination policy for low-performing runs
*	Allocate resources
*	Launch an experiment with the defined configuration

With AutoML 
*	Select experiment type: Classification, Regression, or Time Series Forecasting
*	Data source, formats, and fetch data
*	Choose compute target: local or remote
*	Automated machine learning experiment settings
*	Run an automated machine learning experiment

With Hyperdrive, we must manually choose the right model, the right parameter search space, the right primary metric and the right early termination policy. There are many risks of error and it requires significant architecture engineering. With AutoML, all those risks are skipped and the various stages in the pipeline are automated, this is why AutoML will surely get better performance.

## Future work
On Hyperparameter optimization using HyperDrive, It will be interesting to see if we get better result with Grid or Bayesian parameter sampling method. And also we can choose another primary metric.

AutoML currently supports a bunch of models, it will be interesting to check what are the templates used in the ML models. It will be also interesting to compare the result obtained with Azure AutoML and google AutoML.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
