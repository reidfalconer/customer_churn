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
library(ROCR) # for drawing ROC curves
library(mlr) # for modeling and hyperparameter tuning
library(tidyverse) #load tidyverse last to avoid namespace conflicts
library(parallelMap)


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


df <- df %>% 
  mutate(cluster = cluster %>% 
           factor(levels = class_levels) %>% 
           as.factor())

glimpse(df)

# drop the remaining components that are not needed for the prediction model.
df <- df %>% select(-bcookie, -timestamp)
glimpse(df)


#> Train-test split ----

# use caret to generate a 70-30 train-test-split index
set.seed(90210)
train_index <- createDataPartition(df$cluster, p = 0.7, list = F, times = 1)

# use index to split data into train and test sets
train <- df %>% slice(train_index) 
test <- df %>% slice(-train_index) 


# create mlr tasks
trainTask <- makeClassifTask(data = as.data.frame(train), target = 'cluster')
testTask <- makeClassifTask(data = as.data.frame(test), target = 'cluster')

# HYPERPARAMETER TUNING VIA MLR RANDOM GRID SEARCH ====

# check what xgboost parameters there are and which can be tuned
getParamSet('classif.xgboost')

# list the measures that can be used to tune/train the current model
listMeasures(trainTask)

# Create an xgboost learner that is classification based and outputs
# labels (as opposed to probabilities)
xgb_learner <- makeLearner(
  'classif.xgboost',
  predict.type = 'prob',
  par.vals = list(
    booster = 'gbtree',
    objective = 'binary:logistic',
    eta = 0.1,
    max_depth = 25, 
    min_child_weight = 5, 
    subsample = 0.85, 
    colsample_bytree = 1,
    eval_metric = 'auc',
    nrounds = 200,
    early_stopping_rounds = 100,
    verbose = 1,
    print_every_n = 1
  )
)

# setup a parameter grid to search over when tuning
xgb_params <- makeParamSet(
  makeIntegerParam('max_depth', lower = 5, upper = 40),
  makeNumericParam('subsample', lower = 0.5, upper = 1),
  makeNumericParam('eta', lower = 0.01, upper = 0.1),
  makeIntegerParam('nrounds', lower = 10, upper = 500),
  makeIntegerParam('min_child_weight', lower = 1, upper = 20)
)

# set tune control object to do random searches over the parameter space
control <- makeTuneControlRandom(maxit = 10)
control <- makeTuneControlIrace(maxExperiments = 200L)

# create a 5-fold cv resampling plan
resample_desc <- makeResampleDesc('CV', iters = 10)

parallelStartMulticore(2)
(res = tuned_params <- tuneParams(
  learner = xgb_learner,
  task = trainTask,
  resampling = resample_desc,
  par.set = xgb_params,
  control = control,
  measures = list(auc)
))
parallelStop()

tuning_info <- as.data.frame(res$opt.path)
print((tuning_info[, -ncol(tuning_info)]))

generateHyperParsEffectData(res)

#> Model training ----

model_empl = train(setHyperPars(xgb_learner, par.vals = c(res$x, nthread = 4, verbose = 1)), trainTask)

# model predictions on test data 
pred <- testTask %>% 
  predict(model_empl, .)

test_label <- test$cluster %>% 
  as.factor()

xgbpred <- pred$data$response %>% 
  as.factor()

auc(test_label, xgbpred)


# get feature importance
feature_importance <- getFeatureImportance(model_empl)

# use ggplot to graph 20 most important features
feature_importance$res %>% 
  t() %>% 
  as.data.frame() %>% 
  rownames_to_column() %>% 
  arrange(-V1) %>% 
  rename(importance = V1) %>% 
  top_n(20) %>% 
  mutate(rowname = rowname %>% reorder(importance)) %>% 
  ggplot(aes(y = importance, x = rowname)) + 
  geom_col() +
  coord_flip()



