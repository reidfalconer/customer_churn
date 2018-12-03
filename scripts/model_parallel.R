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
library(readr)

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
  mutate(cluster = cluster %>% 
           factor(levels = class_levels) %>% 
           as.factor())

# drop the remaining components that are not needed for the prediction model.
df <- df %>% select(-bcookie, -timestamp)
glimpse(df)

# TRAIN-TEST SPLIT AND CV FOLDS ====

# use caret to generate a 70-30 train-test-split index
set.seed(8910)
train_index <- createDataPartition(df$cluster, p = 0.70, list = FALSE, times = 1)

# use index to split data into train and test sets
train <- df %>% slice(train_index) 
test <- df %>% slice(-train_index) 

# Define training control. 10-fold cross-validation 
#train_control <- trainControl(method = "cv", number = 10)
cv <- createFolds(train$cluster, k = 10)

# MODELLING ====

formula <- as.formula(paste("cluster ~ ", paste(colnames(train[,1:(length(train)-1)]), collapse= "+")))

xgboostGrid <- expand.grid(#nrounds = 10,
                           nrounds = 1000,
                           #eta = 1,
                           eta = seq(0.1,1,0.1),
                           #gamma = 1,
                           gamma = c(0.8,0.9,1),
                           #colsample_bytree = 1.0,
                           colsample_bytree = c(0.5,0.7,1.0),
                           #max_depth = 6,
                           max_depth = c(4,6,8),
                           #min_child_weight = c(7,8),
                           min_child_weight = c(1,7,8),
                           subsample = 1)

xgboostControl = trainControl(method = "cv",
                              number = 10,
                              classProbs = TRUE,
                              search = "grid",
                              allowParallel = TRUE,
                              summaryFunction=twoClassSummary)

# Set the maximum number of threads to be used.
max.threads <- 4

# Numeric vector for storing the dt for each loop.
tt <- numeric(max.threads)

for (i in seq(1, max.threads, by = 1)) {
  # Invoke the garbage collector.
  gc()
  # Force the system to sleep.
  Sys.sleep(1)
  
  start.time <- as.numeric(as.POSIXct(Sys.time()))
  
  model.training <- train(formula,
                          data = train,
                          method = "xgbTree",
                          trControl = xgboostControl,
                          tuneGrid = xgboostGrid,
                          verbose = TRUE,
                          metric = "ROC",
                          nthread = i)
  
  end.time <- as.numeric(as.POSIXct(Sys.time()))
  tt[i] <- end.time - start.time
  print(paste(c("tt: ", tt[i]), sep = "", collapse = ""))
  
  model.training
  model.training$results
  
  model.test.pred <- predict(model.training, 
                             test %>% select(-cluster), 
                             type = "raw",
                             norm.votes = TRUE)
  
  model.test.prob <- predict(model.training, 
                             test %>% select(-cluster), 
                             type = "prob",
                             norm.votes = TRUE)
  
  performance <- confusionMatrix(model.test.pred, test$cluster)
  print(performance)
}

plot(1:max.threads, 
     tt, 
     type="b",
     frame = FALSE,
     xlab="Number of Cores",
     ylab="Total Time (sec)",
     main="Performance Improvement",
     pch=20, 
     cex=2)

# Invoke the garbage collector one final time.
gc()




