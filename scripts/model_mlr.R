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
library(knitr)  # just using this for kable() to make pretty tables
library(ModelMetrics )

#> Set options ----

# disable scientific notation
options(scipen = 999)


# LOAD DATA & WRANGLE ====

#read data into df 
df <- read_csv("data/churn.csv")

glimpse(df)


ST <- c(ST_act_d1, ST_act_d2, ST_act_d3, ST_act_d4, ST_act_d5, ST_act_d6, ST_act_d7)
dwelltime <- c()


ST_act_d1_mean <0

median(df$sum_ST_cliks)


df_feat <- df %>% 
  mutate(mean_ST_act = rowMeans(select(df, starts_with("ST_act"))),
         mean_ST_num_sessions = rowMeans(select(df, starts_with("ST_num_sessions"))),
         mean_ST_dwelltime = rowMeans(select(df, starts_with("ST_dwelltime"))),
         mean_ST_pageviews = rowMeans(select(df, starts_with("ST_pageviews"))),
         mean_ST_cliks = rowMeans(select(df, starts_with("ST_cliks"))),
         sum_ST_act = rowSums(select(df, starts_with("ST_act"))),
         sum_ST_num_sessions = rowSums(select(df, starts_with("ST_num_sessions"))),
         sum_ST_dwelltime = rowSums(select(df, starts_with("ST_dwelltime"))),
         sum_ST_pageviews = rowSums(select(df, starts_with("ST_pageviews"))),
         sum_ST_cliks = rowSums(select(df, starts_with("ST_cliks"))),
         mean_click_dwell = mean_ST_cliks*mean_ST_dwelltime,
         mean_session_dwell = mean_ST_num_sessions*mean_ST_dwelltime,
         mean_page_dwell = mean_ST_pageviews*mean_ST_dwelltime,
         ST_act_sum_1_3 = rowSums(df %>% select(ST_act_d1, ST_act_d2,ST_act_d3)),
         ST_act_sum_2_4 = rowSums(df %>% select(ST_act_d2, ST_act_d3,ST_act_d4)),
         ST_act_sum_3_5 = rowSums(df %>% select(ST_act_d3, ST_act_d3,ST_act_d5)),
         ST_act_sum_4_6 = rowSums(df %>% select(ST_act_d4, ST_act_d3,ST_act_d6)),
         ST_act_sum_5_7 = rowSums(df %>% select(ST_act_d5, ST_act_d3,ST_act_d7)),
         consecutive_3 = ifelse((ST_act_sum_1_3 == 3 |
                                   ST_act_sum_2_4 == 3 |
                                   ST_act_sum_3_5 == 3 |
                                   ST_act_sum_4_6 == 3 |
                                   ST_act_sum_5_7 == 3), 1, 0),
         click_binary =ifelse(sum_ST_cliks >= median(sum_ST_cliks), 1, 0),
         page_binary =ifelse(sum_ST_pageviews >= median(sum_ST_pageviews), 1, 0),
         dwell_binary =ifelse(sum_ST_dwelltime >= median(sum_ST_dwelltime), 1, 0),
         session_binary =ifelse(sum_ST_num_sessions >= median(sum_ST_num_sessions), 1, 0),
         act_binary =ifelse(sum_ST_act >= median(sum_ST_act), 1, 0)
         ) %>% 
  select(-c(starts_with("ST_act_sum_")))
  
glimpse(df_feat)

corr <- df_feat %>% 
  select(-timestamp, )

# Compute and plot a correlation matrix of remaining predictors.
library(corrplot)
df_feat %>%
  cor(.) %>%
  corrplot(., order = "hclust", tl.cex = .35)

# explicitly define vector of label class levels for reuse and subsequently
# the number of classes for reuse
class_levels <- c('churned', 'engaged')
class_nlevels <- class_levels %>% length()


#c onvert job family categories into numeric variables (necessary for xgboost)
# where: 1 - churned, 2 - engaged
df_feat <- df_feat %>% 
  mutate(cluster_num = cluster %>% 
           factor(levels = class_levels) %>% 
           as.numeric())

df_feat <- df_feat %>% mutate(cluster_num = cluster_num - 1)


df_feat <- df_feat %>%
  mutate_at(
    .vars = vars("cluster_num"),
    .funs = funs(as.factor(.))
  ) 

df_feat <- df_feat %>% select(-bcookie, -timestamp, -cluster)
glimpse(df_feat)


# Feature Normalization
df <- normalizeFeatures(df_feat, target = "cluster_num")
glimpse(df_feat)

# Convert factors to dummy variables
df <- createDummyFeatures(
  df_feat, target = "cluster_num")
glimpse(df_feat)



#> Train-test split ----

# use caret to generate a 70-30 train-test-split index
set.seed(90210)
train_index <- createDataPartition(df_feat$cluster_num, p = 0.6, list = F, times = 1)


# use index to split data into train and test sets
train <- df_feat %>% slice(train_index) 
test <- df_feat %>% slice(-train_index) 

trainTask <- makeClassifTask(data = as.data.frame(train), target = "cluster_num", positive = 1)
testTask <- makeClassifTask(data = as.data.frame(test), target = "cluster_num")




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


xgb_params <- makeParamSet(
  # The number of trees in the model (each one built sequentially)
  makeIntegerParam("nrounds", lower = 100, upper = 1000),
  # number of splits in each tree
  makeIntegerParam("max_depth", lower = 1, upper = 10),
  # "shrinkage" - prevents overfitting
  makeNumericParam("eta", lower = 0.01, upper = 0.5),
  # L2 regularization - prevents overfitting
  #makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x),
  makeNumericParam('subsample', lower = 0.5, upper = 1),
  makeNumericParam("gamma", lower = 0.8, upper = 1),
  makeNumericParam("min_child_weight", lower = 2, upper = 8),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1)
)

# set tune control object to do random searches over the parameter space
control <- makeTuneControlRandom(maxit = 30)

control <- makeTuneControlGrid

# create a 5-fold cv resampling plan
resample_desc <- makeResampleDesc('CV', iters = 10)

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

# Make a new prediction
result <- predict(xgb_model, testTask)

test_label <- test$cluster_num %>% 
  as.factor()

xgbpred <- result$data$response %>% 
  as.factor()

library(caret)
library(pROC)

caret::confusionMatrix(test_label, xgbpred)

auc(as.numeric(test_label), as.numeric(xgbpred))

# Compute AUC for the model.
model.roc <- plot.roc(...)




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







