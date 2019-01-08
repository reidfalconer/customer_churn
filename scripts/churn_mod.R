# METADATA ====
# Description: Simple customer churn model
# Created: 2018-12-03 (Reid Falconer)
# Updated: 2018-12-03 (Reid Falconer)

# INITIALISE ====

rm(list=ls())

#> Libraries ----
library(caret) #train-test splitting
library(xgboost)
library(mlr) # for modeling and hyperparameter tuning
library(tidyverse) #load tidyverse last to avoid namespace conflicts
library(parallelMap)
library(ModelMetrics )
library(pROC)
library(parallel)

#> Set options ----

# disable scientific notation
options(scipen = 999)

# Functions ----

classDistribution <- function(dataset.name = NULL, table = NULL, class = ncol(table))
{    
  print(paste("Class Distribution:", dataset.name, sep = " "));
  print(prop.table(table(table[,class])))
  cat("\n");
}

# LOAD DATA & WRANGLE ====

#read data into df 
df <- read_csv("data/churn.csv")

# Count number of lines in input file.
nrow(df)

# start time
start.time <- as.numeric(as.POSIXct(Sys.time()))

# Remove redundant features.

df <- df %>% select(-bcookie, -timestamp)

# Tranform the class values to factors.
df <- df %>%
  mutate_at(
    .vars = vars("cluster"),
    .funs = funs(as.factor(.))
  ) 

# Feature engineering.
df_feat <- df %>% 
  mutate(mean_ST_act = rowMeans(select(df, starts_with("ST_act"))), # mean active days
         mean_ST_num_sessions = rowMeans(select(df, starts_with("ST_num_sessions"))), # mean sessions
         mean_ST_dwelltime = rowMeans(select(df, starts_with("ST_dwelltime"))), # mean dwell time
         mean_ST_pageviews = rowMeans(select(df, starts_with("ST_pageviews"))), # mean page views
         mean_ST_cliks = rowMeans(select(df, starts_with("ST_cliks"))), # mean clicks
         sum_ST_act = rowSums(select(df, starts_with("ST_act"))), # sum of active days
         sum_ST_num_sessions = rowSums(select(df, starts_with("ST_num_sessions"))), # sum number of sessions
         sum_ST_dwelltime = rowSums(select(df, starts_with("ST_dwelltime"))), # sum dwell time
         sum_ST_pageviews = rowSums(select(df, starts_with("ST_pageviews"))), # sum page views
         sum_ST_cliks = rowSums(select(df, starts_with("ST_cliks"))), # sum number of clicks
         mean_click_dwell = mean_ST_cliks*mean_ST_dwelltime, # interaction between clicks and dwell time
         mean_session_dwell = mean_ST_num_sessions*mean_ST_dwelltime, # interaction between sessions and dwell time
         mean_page_dwell = mean_ST_pageviews*mean_ST_dwelltime, # interaction between page and dwell time
         click_binary = ifelse(sum_ST_cliks >= median(sum_ST_cliks), 1, 0), # sum number of clicks larger than median
         page_binary = ifelse(sum_ST_pageviews >= median(sum_ST_pageviews), 1, 0), # sum pageviews larger than median
         dwell_binary = ifelse(sum_ST_dwelltime >= median(sum_ST_dwelltime), 1, 0), # sum dwell time larger than median
         session_binary = ifelse(sum_ST_num_sessions >= median(sum_ST_num_sessions), 1, 0), # sum sessions larger than median
         act_binary = ifelse(sum_ST_act >= median(sum_ST_act), 1, 0), # sum active days larger than median
         inactive_sessions = 7 - sum_ST_act # num of inactive days
  ) %>% 
  select(-ST_act_d1)

# Other preprocessing
df_feat <- scale(df_feat %>% select(-cluster))
df_feat <- df_feat %>%
  as.data.frame() %>%
  mutate(cluster = df$cluster)

# Perform stratified bootstrapping (keep 60% of observations for training and 40% for testing).
train_index <- createDataPartition(df_feat$cluster, p = 0.60, list = F, times = 1)

# use index to split data into train and test sets
train <- df_feat %>% slice(train_index) 
test <- df_feat %>% slice(-train_index) 

#preparing matrix 
train <- xgb.DMatrix(data = train,label = cluster) 
test <- xgb.DMatrix(data = test,label=cluster)

# Dealing with class Imbalance ----

# Creates a sample of synthetic data by enlarging the features space of 
# minority and majority class examples. Operationally, the new examples 
# are drawn from a conditional kernel density estimate of the two classes, 
# as described in Menardi and Torelli (2013).

# library(ROSE)
# table(train$cluster)
# train <- ROSE(cluster ~., train)$data
# table(train$cluster)

# this procedure did not yeild improved results and was subsequently abandoned.


# Make train and text tasks for the classifier
trainTask <- makeClassifTask(data = as.data.frame(train), target = "cluster", positive = "churned")
testTask <- makeClassifTask(data = as.data.frame(test), target = "cluster")

# Compute class distribution.
classDistribution(dataset.name = "df",
                  table = df,
                  class = length(df))

classDistribution(dataset.name = "train",
                  table = train,
                  class = length(train))

classDistribution(dataset.name = "test",
                  table = test,
                  class = length(test))

# HYPERPARAMETER TUNING VIA MLR RANDOM GRID SEARCH ====

# check what xgboost parameters there are and which can be tuned
getParamSet('classif.xgboost')

# list the measures that can be used to tune/train the current model
listMeasures(trainTask)

# Create an xgboost learner that is classification based and outputs
# default parameters
xgb_learner <- makeLearner(
  'classif.xgboost',
  predict.type = 'prob',
  par.vals = list(
    booster = 'gbtree',
    objective = 'binary:logistic',
    eta = 0.1,
    max_depth = 5, 
    min_child_weight = 5, 
    subsample = 0.85, 
    colsample_bytree = 1,
    eval_metric = 'auc',
    nrounds = 500,
    early_stopping_rounds = 50,
    verbose = 1,
    print_every_n = 1
  )
)

#set parameter space
xgb_params <- makeParamSet(
  # The number of trees in the model (each one built sequentially)
  makeIntegerParam("nrounds", lower = 100, upper = 200),
  # number of splits in each tree
  makeIntegerParam("max_depth", lower = 1, upper = 2),
  # "shrinkage" - prevents overfitting
  makeNumericParam("eta", lower = 0.07, upper = 0.1),
  # L2 regularization - prevents overfitting
  #makeNumericParam("lambda", lower = -1, upper = -.1, trafo = function(x) 10^x),
  makeNumericParam('subsample', lower = .8, upper = 1),
  makeNumericParam("gamma", lower = 0.7, upper = 1),
  makeNumericParam("min_child_weight", lower = 4, upper = 10),
  makeNumericParam("colsample_bytree", lower = 0.7, upper = 1)
)

# set tune control object to do random searches over the parameter space
control <- makeTuneControlRandom(maxit = 2)

# create a 10-fold cv resampling plan
resample_desc <- makeResampleDesc('CV', stratify = TRUE, iters = 10)

#set parallel backend
parallelStartMulticore()
(tuned_params <- tuneParams(
  learner = xgb_learner,
  task = trainTask,
  resampling = resample_desc,
  par.set = xgb_params,
  control = control,
  measures = list(mlr::auc),
  show.info = TRUE
))
parallelStop()

# extract tuned hyperparameters
generateHyperParsEffectData(tuned_params, partial.dep = TRUE)
tuning_info <- as.data.frame(tuned_params$opt.path)
print((tuning_info[, -ncol(tuning_info)]))

# Create a new model using tuned hyperparameters
xgb_tuned_learner <- setHyperPars(
  learner = xgb_learner,
  par.vals = tuned_params$x
)

# Re-train parameters using tuned hyperparameters (and full training set)
xgb_model <- train(xgb_tuned_learner, trainTask)

# Print basic information about our model.
xgb_model

# Evaluate model on test set.
result <- predict(xgb_model, testTask)

end.time <- as.numeric(as.POSIXct(Sys.time()))
time <- end.time - start.time
print(time)

# Compute confusion matrix.

test_label <- test$cluster 
xgbpred <- result$data$response

caret::confusionMatrix(test_label, xgbpred)

# Compute AUC for the model.
model.roc <- roc(test_label %>% as.numeric(), xgbpred %>% as.numeric(), auc.polygon=TRUE, grid=TRUE, plot=FALSE)
model.roc
