# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:40:49 2019

@author: Mohammad Ehtasham Billah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier



# Loading the data
df = pd.read_csv("Churn_Modelling.csv")
df.head()
df.shape
df.dtypes
df.describe()
df.corr()
df.skew()

# Data preprocessing
# Dummy varibale
cat_features = ["Geography","Gender"]
df_final = pd.get_dummies(data = df,columns = cat_features,drop_first = True)


# Scalling
x = df_final.iloc[:,3:].drop("Exited",axis = 1)
y = df_final["Exited"]
scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)
scaled_x = pd.DataFrame(data = scaled_x,columns=x.columns)

# Splitting the data
test_size = 0.20
seed = 2019
x_train,x_test,y_train,y_test = train_test_split(scaled_x, y, 
                                                 test_size = test_size,
                                                 random_state = seed)

# k-fold cross validation with gridsearch
# 1. Parameter grid 
optimizer= ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
kernel_initializer= ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
dropout_rate = [0.0,0.1,0.2,0.3,0.4,0.5]
weight_constraint= [1,2,3,4,5]
units = [1,6,11,16,21,26,31]


# 2. Hyperparameter grid
batch_size = [25,32]
epochs = [100,200,500]


param_grid = dict(batch_size = batch_size,
                  epochs = epochs,
                  optimizer = optimizer,
                  kernel_initializer = kernel_initializer,
                  activation = activation,
                  dropout_rate = dropout_rate,
                  weight_constraint = weight_constraint,
                  units = units)


# Function for cross validation with grid search
def create_classifier(optimizer,kernel_initializer,activation,dropout_rate,weight_constraint,units):
    ann_classifier = Sequential()
    ann_classifier.add(Dense(input_dim = 11,              # no. of nodes in input layer = number of independent variables
                         units = units,                                     # gridsearching
                         kernel_initializer= kernel_initializer,            # gridsearching
                         activation= activation ,                           # gridsearching        
                         kernel_constraint = maxnorm(weight_constraint)     # gridsearching
                         ))
    ann_classifier.add(Dropout(dropout_rate))                               # droping out units from 1st hidden layer
    ann_classifier.add(Dense(units = units,                                 # gridsearching    
                         kernel_initializer= kernel_initializer,            # gridsearching
                         activation= activation,                            # gridsearching
                         kernel_constraint = maxnorm(weight_constraint)     # gridsearching
                         ))
    ann_classifier.add(Dropout(dropout_rate))                               # droping out units from 1st hidden layer
    ann_classifier.add(Dense(units = 1,                                     # no of units in output layer for 2-class classification (p=yes/poitive/1)
                         kernel_initializer= kernel_initializer,            # gridsearching
                         activation= "sigmoid"
                         ))
    ann_classifier.compile(optimizer = optimizer,                           # gridsearching
                       loss = "binary_crossentropy",       # loss function we want to minimize for 2-class classification
                       metrics = ["accuracy"]
                       )
    return ann_classifier

ann_classifier = KerasClassifier(build_fn = create_classifier)

# visualization of the model
print(ann_classifier.summary())
plot_model(ann_classifier, to_file='ann_classifier_plot.png', show_shapes=True, show_layer_names=True)

kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)
grid = GridSearchCV(estimator = ann_classifier,
                    param_grid = param_grid,
                    scoring = "accuracy",
                    cv = kfold,
                    verbose = 1)

# Fitting the model
grid_results = grid.fit(x_train,y_train)


best_accuracy = grid.best_score_
best_parameters = grid.best_params_

print("Best {} using {}".format(best_accuracy,best_parameters))
cv_means = grid_results.cv_results_["mean_test_score"]
cv_stds = grid_results.cv_results_["std_test_score"]
cv_params = grid_results.cv_results_["params"]

for mean,std,param in zip(cv_means,cv_stds,cv_params):
    print("{} ({}) with {}".format(mean,std,param))


