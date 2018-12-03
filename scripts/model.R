# METADATA ====
# Description: First attempt at a simple customer churn model
# Created: 2018-12-03 (Reid Falconer)
# Updated: 2018-12-03 (Reid Falconer)
# Reviewed: 

# INITIALISE ====

rm(list=ls())

#> Libraries ----
library(caret) #train-test splitting
library(xgboost)
library(tidyverse) #load tidyverse last to avoid namespace conflicts
library(ModelMetrics)

#> Set options ----

# disable scientific notation
options(scipen = 999)


# LOAD DATA & WRANGLE ====

#read data into df 
df <- read_csv("data/churn.csv")

glimpse(df)

# explicitly define vector of label class levels for reuse and subsequently
# the number of classes for reuse
class_levels <- c('churned', 'engaged')
class_nlevels <- class_levels %>% length()

#c onvert job family categories into numeric variables (necessary for xgboost)
# where: 1 - churned, 2 - engaged
df <- df %>% 
  mutate(cluster_num = cluster %>% 
           factor(levels = class_levels) %>% 
           as.numeric())

# the XGBoost algorithm requires that the class labels (job family) start at 0 and increase sequentially 
# to the maximum number of classes. This is a bit of an inconvenience as you need to keep track of what job 
# family name goes with which label. Also, you need to be very careful when you add or remove a 1 to go from 
# the zero based labels to the 1 based labels. Outcome:
# 0 - ana, 1 - ass, 2 - cre, 3 - dri, 4 - ele, 5 - pro, 6 - sss
df <- df %>% mutate(cluster_num = cluster_num - 1)

# drop the remaining components that are not needed for the prediction model.
df <- df %>% select(-bcookie, -cluster, -timestamp)
glimpse(df)

# TRAIN-TEST SPLIT AND CV FOLDS ====

# use caret to generate a 70-30 train-test-split index
set.seed(8910)
train_index <- createDataPartition(df$cluster_num, p = 0.70, list = FALSE, times = 1)

# use index to split data into train and test sets
train <- df %>% slice(train_index) 
test <- df %>% slice(-train_index) 

# Define training control. 10-fold cross-validation 
#train_control <- trainControl(method = "cv", number = 10)
cv <- createFolds(train$cluster_num, k = 10)

# MODELLING ====

train_mat <- train  %>%  
  select(-cluster_num) %>% 
  as.matrix()

test_mat <- test  %>%  
  select(-cluster_num) %>% 
  as.matrix()

#> Preprocessing ----

# prepare the train and test matrices for xgboost.
dtrain <- xgb.DMatrix(data = train_mat, label = train$cluster_num)
dtest <- xgb.DMatrix(data =  test_mat, label = test$cluster_num)

gc()


#> Grid Search ----





#> Cross validation ----

# set xgboost parameters (still need to go through parameter tuning - this is just default first attempt)
params <- list(booster = 'gbtree', 
               objective = 'binary:logistic',
               eta = 0.3, 
               gamma = 0, 
               max_depth = 6, 
               min_child_weight = 1, 
               subsample = 0.8)

# preform cross-validated xgboost to get the best iteration
xgboost_cv <- xgb.cv(param = params, 
                     data = dtrain, 
                     folds = cv, 
                     nrounds = 800, 
                     early_stopping_rounds = 100,
                     metrics = 'auc',
                     verbose = TRUE,
                     prediction = TRUE)

# best_iteration
best_iteration = xgboost_cv$best_iteration


#> Train model ----

# fit xgboost model on training data
mod_xgb <- xgb.train(data = dtrain, 
                     params = params,
                     nrounds = best_iteration,  
                     eval_metric = 'auc')

# load xgb model form file path
#mod_xgb <- xgb.load("models/xgb_job_family_classifier_v0.0.0.RDS")
mod_xgb


# POSTESTIMATION ====

# Predict hold-out test set. 
xgbpred <- predict (mod_xgb,dtest)

xgbpred <- ifelse (xgbpred > 0.5,1,0) %>% 
  as.factor()
test_label <- test$cluster_num %>% 
  as.factor()

# confusion matrix of test set
confusionMatrix(xgbpred, test_label)


auc(test_label, xgbpred)




#> Feature importance ----

# get the trained model.
model <- xgb.dump(mod_xgb, with_stats = TRUE)

# get the feature real names.
names <- dimnames(dtrain)[[2]]

# compute feature importance matrix
importance_matrix <- xgb.importance(names, model = mod_xgb)

# plot feature importance maxtrix for top 30 most important features
xgb.plot.importance(importance_matrix[0:10] )


# EXPORT MODEL ====

# save the model for future using
#xgb.save(mod_xgb,"models/xgb_job_family_classifier_v0.0.0.RDS")


