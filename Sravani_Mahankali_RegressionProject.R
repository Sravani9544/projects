---
title: "R Notebook"
output: html_notebook
---

#______________________________________________________________________________
#*Regression analysis*#
#______________________________________________________________________________

## installing the necessary packages 


install.packages("tidyr")
install.packages("caret")
install.packages("data.table")
install.packages("doParallel")

library(tidyr)
library(caret)
library(data.table)
library(MASS)
library(class)
library(doParallel)
library(gbm)
library(randomForest)
library(dplyr)
library(Metrics)
library(glmnet)



## loading the data set


lifeExpectancy_data <- read.csv(file = "Life Expectancy Data.csv")


## summary of the data set 


summary(lifeExpectancy_data)


#* Observation: There are NAs in some of the columns and there is need to clean
#   those

#______________________________________________________________________________

### prepocessing data set 

## converting variables to factors 


lifeExpectancy_data$Country <- as.factor(lifeExpectancy_data$Country)
lifeExpectancy_data$Status <- as.factor(lifeExpectancy_data$Status)


## removing NAs from the dataset


life_data <- lifeExpectancy_data %>% drop_na(Income.composition.of.resources)%>%
  drop_na(Life.expectancy)%>%
  drop_na(Adult.Mortality)%>%
  drop_na(Alcohol)%>%
  drop_na(Hepatitis.B)%>%
  drop_na(BMI)%>%
  drop_na(Polio)%>%
  drop_na(Total.expenditure)%>%
  drop_na(Diphtheria)%>%
  drop_na(GDP)%>%
  drop_na(Population)%>%
  drop_na(thinness..1.19.years)%>%
  drop_na(thinness.5.9.years)%>%
  drop_na(Schooling)


## checking the summary again


summary(life_data)

View(life_data)


## checking for zero/ non zero producing variables


nzv1 <- nearZeroVar(life_data, saveMetrics= TRUE)



nzv1[nzv1$nzv1,]


#* Observation: There are no zero variance / non zero variance producing variables



 dim(life_data)


## storing remaining variables in a data set 


nzv1 <- nearZeroVar(life_data)
filteredDescrLife <- life_data[, -(nzv1)]
dim(filteredDescrLife)



## shuffing the data set 


for (iternum in (1:3)){
  print(paste("Shuffle number:", iternum))
  housingdataset <- life_data[sample(nrow(life_data)), ]
}


## partitioning the data set into 75 % training data set, and 25% holdout set 


set.seed(2356)
inTrainingLife <- createDataPartition(life_data$Life.expectancy, 
                                  p = .75, 
                                  list = FALSE)
trainingLife <- life_data[ inTrainingLife,]
holdoutLife  <- life_data[-inTrainingLife,]


## scaling the training data set



life.training <- as.data.frame(sapply(trainingLife[-c(1,3,4)], scale))


## scaling the holdout data set




life.holdout <- as.data.frame(sapply(holdoutLife[-c(1,3,4)], scale))



## creating a data set of training dataset and outcome variable



life.reduced <- data.frame(life.training, trainingLife$Life.expectancy)
names(life.reduced) <- c(names(life.reduced[1:19]), "Life.expectancy")



## creating a data set of holdout dataset and outcome variable


life.holdout <- data.frame(life.holdout, holdoutLife$Life.expectancy)
names(life.holdout) <- c(names(life.holdout[1:19]), "Life.expectancy")


## producing cross validaton for all the models


fitControl <- trainControl(method = "repeatedcv", 
                           number = 6, 
                           repeats = 6,
                           returnResamp = "final", 
                           selectionFunction = "best")


## performing various modelling techniques

# 1. Multiple Linear Regression


## fitting the training data set 

set.seed(2356)
mlr.Fit <- train(Life.expectancy ~ .,
                 data = life.reduced, 
                 method = "glm",
                 metric = 'RMSE',
                 trControl = fitControl)
                
mlr.Fit

## checking the summary of the MLR model

summary(mlr.Fit)

## computing predictions on holdout set 

predvals.mlr <- predict(mlr.Fit, life.holdout)

## computing holdout metrics 

postResample(pred = predvals.mlr, obs = life.holdout$Life.expectancy)

## computing variable importance

varImp(mlr.Fit)


#* Observation:  The Root Mean Square error of the multiple linear regression 
#  model on holdout set is 3.66 (approx). The most important variables as 
#  observed in the model summary and while computing variable importance are 
#  HIV.AIDS[p-value = < 2e-16], Adult.Mortality[p-value = < 2e-16], Schooling 
#  [p-value = < 2e-16], Income.composiiton.of.resources [p-value = < 2e-16], 
#  under.five.death[p-value = 1.27e-12], infant.deaths[p-value = 5.02e-12], 
#  year[p-value = 1.54e-07], and BMI[p-value = 1.08e-06],Alcohol[p-value = 0.0265], 
#  percentage.expenditure[p-value = 0.0328] are less important. However, all other 
#  variables are not important.

# 2. Lasso Regression




## creating a grid with tuning parameters

set.seed (2356)
lassoGrid <-  expand.grid(alpha = 0.1, lambda = 1 )

nrow(lassoGrid)
lassoGrid

## fitting the model on training dats set

set.seed(2356)
lasso.Fit <- train(Life.expectancy ~ .,
                 data = life.reduced, 
                 method = "glmnet", 
                 metric = 'RMSE',
                 trControl = fitControl,
                 tuneGrid = lassoGrid )
                
lasso.Fit

## checking the summary of the model

summary(lasso.Fit)

## predictions on holdout set

predvals.lasso <- predict(lasso.Fit, life.holdout)

## computing metrics for the hold out set 

postResample(pred = predvals.lasso, obs = life.holdout$Life.expectancy)

## computing the variable importance

varImp(lasso.Fit)


#*Observation: The Root mean square error produced by the model on holdout set
# is 3.8420078. The most important variables predicted by the lasso method are 
# HIV.AIDS, Adult.Mortality, Schooling, Income.compensation.of.resources, BMI, 
# percentage.expenditure, GDP, Year, Diphteria.

# 3. Generalised Additive Models



## hyperparameter tuning grid for model

set.seed (2356)
gamGrid <-  expand.grid(df = c(1:6))

nrow(gamGrid)
gamGrid

## fitting random forest model on training data set 

set.seed(2356)
gam.Fit <- train(Life.expectancy ~ .,
                 data = life.reduced, 
                 method = "gamSpline", 
                 trControl = fitControl,
                 metric = 'RMSE',
                 tuneGrid = gamGrid)
                
gam.Fit


## computing predictions on holdout set

predvals.gam <- predict(gam.Fit, life.holdout)

## computing metrics on holdout set 

postResample(pred = predvals.gam, obs = life.holdout$Life.expectancy)

## computing the variable importance

varImp(gam.Fit)



#* Observation: The optimal value of the degree of freedom which produced lower 
#  Root Mean Square Error is 6. The Root Mean square error on training set is 2.568817 
#  and the Root Mean Square Error on holdout set is 2.568817. However, the model
#  did not predict the importance of variables.

# 4. Random Forest


## hyperparameter tuning grid for model

set.seed (2356)
rfGrid <-  expand.grid(mtry = seq(1:10) )

nrow(rfGrid)
rfGrid

## fitting random forest model on training data set

set.seed(2356)
rf.Fit2 <- train(Life.expectancy ~ .,
                 data = life.reduced, 
                 method = "rf", 
                 trControl = fitControl,
                metric = 'RMSE',
                 verbose = FALSE, 
                 tuneGrid = rfGrid)
                
rf.Fit2

summary(rf.Fit2) 

## making predictions on holdout set

predvals.rf2 <- predict(rf.Fit2, life.holdout)

## computing metrics on holdout set

postResample(pred = predvals.rf2, obs = life.holdout$Life.expectancy)

## computing the variable importance

varImp(rf.Fit2)


#* Observation: The optimal mtry value which produced lower Root Mean Square 
#  Error value is 10. The RMSE value on training set is 1.881067 and the RMSE 
#  value on holdout set is 2.0670089.The important variables as predicted by 
#  the model are Income.composition.of.resources, HIV.AIDS, Adult.Mortality, 
#  Schooling, BMI, thinness.5.9.years, thinness..1.19.years, under.five.deaths,
#  Alcohol, Total.expenditure, respectively.

# 5. Boosted Tree


## hyperparameter tuning grid for model

set.seed (2356)
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 10)
nrow(gbmGrid)
gbmGrid


## fitting boosted tree model on training data set

set.seed(2356)
gbmFit <- train(Life.expectancy ~ .,
                 data = life.reduced, 
                 method = "gbm", 
                 trControl = fitControl, 
                 metric = 'RMSE',
                 verbose = FALSE, 
                 tuneGrid = gbmGrid)
gbmFit

 

## checking the summary

summary(gbmFit)

## predictions on holdout set 

predvals.gbm1 <- predict(gbmFit, life.holdout)

## computing metrics for the model

postResample(pred = predvals.gbm1, obs = life.holdout$Life.expectancy)

## variable importance

varImp(gbmFit)


#* Observation: The final values used for the model were n.trees = 1500, interaction.depth = 9,
#  shrinkage = 0.1 and n.minobsinnode = 10 which produced a lower RMSE of 1.929111.The 
#  RMSE produced on holdout set is 2.0710561. The important variables are 
#  respectively, Income.composition.of.resources, HIV.AIDS, Adult.Mortality, 
#  thinness.5.9.years, Total.expenditure, Alcohol, thinness..1.19.years, 
#  percentage.expenditure,Schooling, under.five.deaths, etc.

# 6. Support Vector Machines


## creating the tuning grid for the model

grid <- expand.grid(C = c(0.01, 0.1, 10, 100, 1000))




## fitting SVM model on training data set 

svmFit <- train(Life.expectancy ~ .,
                      data = life.reduced,
                      method = "svmLinear",
                      trControl = fitControl,
                      metric = 'RMSE', 
                      verbose = FALSE,
                      tuneGrid = grid)

svmFit

## checking the summary of the model

summary(svmFit)

## making predictions on the holdout set 

predvals.svm1 <- predict(svmFit, life.holdout)

## computing metrics on holdout set

postResample(pred = predvals.svm1, obs = life.holdout$Life.expectancy)

## variable importance

varImp(svmFit)


#* Observation: The optimal value of c used in the model is c = 100 which produced 
#  lowest RMSE. The RMSE value of model on training set is 3.828355 and the RMSE
#  value of holdout set is 3.7157560. The importance of variables are respectively,
#  Income.composition.of.resources, Adult.Mortality, Schooling, BMI, HIV.AIDS, 
#  percentage.expenditure, GDP, thinness..1.19.years, thinness.5.9.years, 
#  Alcohol, etc.

#______________________________________________________________________________

# comparision of all the models


## resampling all the models for comparision

resamps_life <- resamples(x = list(MLR = mlr.Fit,
                          LASSO = lasso.Fit,
                          GAM = gam.Fit,
                          RF = rf.Fit2,
                          GBM = gbmFit,
                          SVM = svmFit))
resamps

## checking the summary

summary(resamps_life)


#* Observation: According to the Summary of the resamples the Random Forest model
#  has lower RMSE [1.881067] than any other model


## plotting Box-Whiskers plot

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps_life, layout = c(3, 1))

## plotting the dot plot

trellis.par.set(caretTheme())
dotplot(resamps_life, metric = "RMSE")

## plotting density plots of accuracy

scales <- list(x=list(relation="free"), y=list(relation="free"))
densityplot(resamps_life, scales=scales, pch = "|")


#* Observation: All the models produced RMSE lesser than 4. Boosted Trees
#  [1.929111] and Random Forest[1.881067] models have lesser RMSE than all 
#  the other models. After this the Support Vector Machines[2.568817] has lower
# RMSE.



## Computing estimates of difference and p-value for all the models

life_diffs <- diff(resamps_life)

# summarize p-values for pair-wise comparisons

summary(life_diffs)


#* Observation: The Lower diagonal of the table represents the P-value for 
#  knowing the difference between the baseline Mulitiple Linear Regression 
#  Regression Model and the other models. The upper diagonal represents the 
#  estimates of the difference between all the models. All the models has the 
#  p-value less than 0.05 so all the models are good but the estimates of 
#  difference tells that the Random Forests Model is better as it has 0.68 value
#  which represents the amount of the differnce in Random Forest compared to all 
#  the other models.  
