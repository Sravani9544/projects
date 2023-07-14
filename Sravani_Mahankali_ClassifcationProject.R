---
title: "R Notebook"
output: html_notebook
---

#______________________________________________________________________________
#classification analysis#
#______________________________________________________________________________

## installing the necessary packages


install.packages("nnet")
install.packages("nnet")
install.packages("doParallel")
install.packages("tidyr")
install.packages("caret")
install.packages("data.table")



library(tidyr)
library(caret)
library(data.table)
library(MASS)
library(class)
library(doParallel)
library(gbm)
library(randomForest)
library(dplyr)
library(nnet)
library(Metrics)



#______________________________________________________________________________
## Importing the data set

fetalHealth_data <- read.csv(file = "fetal_health.csv")

## checking the summary of the dataset

summary(fetalHealth_data)


# * observation: fetal_health(outcome variable) and histogram_tendency 
#   should be factors

#______________________________________________________________________________

### Pre- Processing the data set 

## converting variables into factors


fetalHealth_data$fetal_health <- as.factor(fetalHealth_data$fetal_health)
fetalHealth_data$histogram_tendency <- as.factor(fetalHealth_data$histogram_tendency)

## checking the summary 


summary(fetalHealth_data)

View(fetalHealth_data)


## checking for zero variance / near zero variance producing variables

nzv <- nearZeroVar(fetalHealth_data, saveMetrics= TRUE)



## printing the variables

nzv[nzv$nzv,]

#* Observation: the sever_decelerations, prolonged_decelerations, 
#  percentage_of_time_with_abnormal_long_term_variability produced zero 
#  variance in the dataset


## checking the dimensions of the original data set 

 dim(fetalHealth_data)

 

## creating the data set after deleting the above variables

nzv <- nearZeroVar(fetalHealth_data)
filteredDescr <- fetalHealth_data[, -nzv]
dim(filteredDescr)




## checking the names of the variables in the remaining dataset

names(filteredDescr)


## shuffling the dataset


for (iternum in (1:3)){
  print(paste("Shuffle number:", iternum))
  housingdataset <- filteredDescr[sample(nrow(filteredDescr)), ]
}


## dividng the dataset into 75% training data set and 25% holdout dataset


set.seed(2356)
inTraining <- createDataPartition(filteredDescr$fetal_health, 
                                  p = .75, 
                                  list = FALSE)
training <- filteredDescr[ inTraining,]
holdout  <- filteredDescr[-inTraining,]


## scaling the training data set by removing the outcome variable and 
## another factor variable(histogram_tendency)


fetal.training <- as.data.frame(sapply(training[-c(18,19)], scale))


## scaling the holdout data set by removing the outcome variable and 
## other factor variable(histogram_tendency)



fetal.holdout <- as.data.frame(sapply(holdout[-c(18,19)], scale))


## creating a data set of training dataset and outcome variable



fetal.reduced <- data.frame(fetal.training, training$fetal_health)
names(fetal.reduced) <- c(names(fetal.reduced[1:17]), "fetal_health")


## creating a data set of holdout dataset and outcome variable


fetal.holdout <- data.frame(fetal.holdout, holdout$fetal_health)
names(fetal.holdout) <- c(names(fetal.holdout[1:17]), "fetal_health")


## producing cross validaton for all the models


fitControl <- trainControl(method = "repeatedcv", 
                           number = 6, 
                           repeats = 6,
                           returnResamp = "final", 
                           selectionFunction = "best")



## Performing various modelling techniques

# 1 Multinomial Regression



## hyperparameter tuning grid for model

set.seed (2356)
multinomGrid <-  expand.grid(decay = c(1:6) )

nrow(multinomGrid)
multinomGrid

## fitting the model on training data set

set.seed(2356)
multinom.Fit <- train(fetal_health ~ .,
                 data = fetal.reduced, 
                 method = "multinom", 
                 metric = 'Accuracy',
                 verbose = FALSE,
                 trControl = fitControl, 
                 tuneGrid = multinomGrid)

summary(multinom.Fit) 

## computing accuracy for model

conf.matrix <- table(fetal.holdout$fetal_health,
                     predict(multinom.Fit, newdata = fetal.holdout))
pred.accuracy <- sum(diag(conf.matrix))/sum(conf.matrix) * 100
pred.accuracy

## computing variable importance 

varImp(multinom.Fit)




#* Observation: The prediction accuracy of the model is 90.188 on holdout.
#  The variable accelerations is most important for predicting the fetal_health
#  stage, followed by abnormal_short_term_variability, histogram_variance,
#  baseline.value, histogram_mean with more than half i.e., 50 overall importance metric. 

# 2. Linear Discriminant Analysis



## fitting the model on training data set

set.seed(2356)
lda.Fit <- train(fetal_health ~ .,
                 data = fetal.reduced, 
                 method = "lda", 
                 metric = 'Accuracy',
                 trControl = fitControl, 
                 verbose = FALSE
                 )
                
lda.Fit

summary(lda.Fit) 

## computing accuracy of model on holdout set

conf.matrix <- table(fetal.holdout$fetal_health,
                     predict(lda.Fit, newdata = fetal.holdout))
pred.accuracy <- sum(diag(conf.matrix))/sum(conf.matrix) * 100
pred.accuracy



## computing variable importance

varImp(lda.Fit)



#* Observation: the prediction accuracy of the model on holdout set is 89.433. 
#  The most important variable with respect to LDA moedl is mean_value_of_short_term_variability, 
#  followed by histogram_median, histogram_mode, abnormal_short_term_variability, 
#  mean_valueof_long_term_variability, accelerations, baseline.value, etc. 

# 3. K- Nearest Neighbors


## fitting the model on training data set

set.seed(2356)
knn.Fit <- train(fetal_health ~ .,
                 data = fetal.reduced, 
                 method = "knn",
                 metric= 'Accuracy',
                 trControl = fitControl,
                 preProcess = c("center","scale"), 
                 tuneLength = 20)
                
knn.Fit

summary(knn.Fit) 

## computing accuracy of model on holdout set

conf.matrix <- table(fetal.holdout$fetal_health,
                     predict(knn.Fit, newdata = fetal.holdout))
pred.accuracy <- sum(diag(conf.matrix))/sum(conf.matrix) * 100
pred.accuracy

## computing variable importance

varImp(knn.Fit)


#* Observation: The accuracy was largest for model with 0.8966165 where k 
#  is equal to 5 and The accuracy on holdout subset is 89.81132. The most 
#  important variable with respect to KNN moedl is mean_value_of_short_term_variability, 
#  followed by histogram_median, histogram_mode, abnormal_short_term_variability,
#  mean_valueof_long_term_variability, accelerations, baseline.value, etc.

# 4. Random Forest



## hyperparameter tuning grid for model

set.seed (2356)
rfGrid <-  expand.grid(mtry = seq(1:6) )

nrow(rfGrid)

## fitting random forest model on training data set

set.seed(2356)
rf.Fit <- train(fetal_health ~ .,
                 data = fetal.reduced, 
                 method = "rf", 
                 metric = 'Accuracy',
                 trControl = fitControl, 
                 verbose = FALSE, 
                 tuneGrid = rfGrid)
                
rf.Fit

summary(rf.Fit) 

## computing accuracy of model on holdout set

conf.matrix <- table(fetal.holdout$fetal_health,
                     predict(rf.Fit, newdata = fetal.holdout))
pred.accuracy <- sum(diag(conf.matrix))/sum(conf.matrix) * 100
pred.accuracy

## computing variable importance

varImp(rf.Fit)



#*Observation: The accuracy of the model on the holdout set was higher The 
# accuracy was largest with 0.9379699 for mtry equal to 5. The accuracy of the 
# model on the holout set is 92.83. The most important variable with respect to 
# Random Forest model is mean_value_of_short_term_variability, followed by 
# histogram_median, histogram_mode, abnormal_short_term_variability, 
# mean_valueof_long_term_variability, accelerations, baseline.value, etc.


# 5. Boosted Tree


## hyperparameter tuning grid for model

set.seed (2356)
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 10)
nrow(gbmGrid)
gbmGrid

## fitting boosted tree model on training dataset

set.seed(2356)
gbmFit2 <- train(fetal_health ~ .,
                 data = fetal.reduced, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE,
                 metric = 'Accuracy',
                 tuneGrid = gbmGrid)
gbmFit2


## computing accuracy of boosted tree model on holdout set

conf.matrix.boosted <- table(fetal.holdout$fetal_health,
                     predict(gbmFit2, newdata = fetal.holdout))
pred.accuracy.boosted <- sum(diag(conf.matrix))/sum(conf.matrix) * 100
pred.accuracy.boosted


## computing variable importance

varImp(gbmFit2)


#* Observation: The accuracy(0.9385965) of the model on traing set was higher 
#  at n.trees = 200,interaction.depth = 9, shrinkage = 0.1 and n.minobsinnode = 10. 
#  The accuracy of Boosted trees on holdout set is 92.83%. The important 
# variables predicted by the model are abnormal_short_term_variability, 
# histogram_mean, mean_value_of_short_term_variability, accelerations, 
# histogram_mode, mean_value_of_long_term_variability, baseline.value.

# 6. Support Vector Machines


## tuning parameter grid for the model

grid <- expand.grid(C = c(0.01, 0.1, 10, 100, 1000))

## fitting the model on training dataset

svmlinearfit <- train(fetal_health ~ .,
                    data = fetal.reduced,
                    method = "svmLinear",
                    trControl = fitControl,
                    metric = 'Accuracy',
                    verbose = FALSE,
                    tuneGrid = grid)

summary(svmlinearfit)


## producing information on the model

names(svmlinearfit)

## computing accuracy of model on holdout set

conf.matrix.svm <- table(fetal.holdout$fetal_health,
                     predict(svmlinearfit, newdata = fetal.holdout))
pred.accuracy.svm <- sum(diag(conf.matrix))/sum(conf.matrix) * 100
pred.accuracy.svm

##  variable importance

varImp(svmlinearfit)


#* Observation: The accuracy of the model on the holdout set is 92.83. The 
#  important variables predicted by the model are mean_value_of_short_term_variability,
#  histogram_mean, histogram_median, histogram_mode,  abnormal_short_term_variability, 
#  mean_value_of_long_term_variability, accelerations, baseline.value, etc. 

#_______________________________________________________________________________

# Comparision of all the models


## resampling all the models for comparision

resamps <- resamples(x = list(MR = multinom.Fit,
                          LDA = lda.Fit,
                          KNN = knn.Fit,
                          RF = rf.Fit,
                          GBM = gbmFit2,
                          SVM = svmlinearfit))
resamps

## checking the summary

summary(resamps)


#* Observation: According to the Summary of the resamples the Gradient Boosted 
#  Tree model has more accuracy[0.9399541] than any other model

## plotting box-whiskers plot

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps, layout = c(3, 1))

## plotting dot plot

trellis.par.set(caretTheme())
dotplot(resamps, metric = "Accuracy")

# plotting density plots of accuracy

scales <- list(x=list(relation="free"), y=list(relation="free"))
densityplot(resamps, scales=scales, pch = "|")


#* Observation: All the models produced accuracy greater than 0.85.Boosted Trees
#  [>0.95] and Random Forest[>0.95] models have greater accuracy than all the
#  other models. After this the Support Vector Machines[>0.89] and K-Nearest 
#  Neighbors [>0.89] have more accuracy.


## checking the estimates of difference and p-value

fetal_diffs <- diff(resamps)

# summarize p-values for pair-wise comparisons

summary(fetal_diffs)


#* Observation: The Lower diagonal of the table represents the P-value for 
#  knowing the difference between the baseline Multinomial Regression Model and
#  the other models. The upper diagonal represents the estimates of the 
# difference between the multinomial model and other models. There is no 
# difference between Multinomial Regression and Gradient Boosted Model and 
# there is no P-value. The SVM Model is better in terms of p-value[< 2.2e-16]
# as it is lower than all the other models. 



