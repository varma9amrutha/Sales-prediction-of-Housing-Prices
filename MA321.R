#Loading required packages
library('ggplot2')
library('tidyverse')
library('scales')
library('dplyr') 
library('caret')
library('MASS')
library('randomForest')

#Load the dataset
setwd('/Users/ankitrajsingh/Desktop/MA321 group')
data<-read.csv('house-data.csv')

#Task 1: Summary statistics and visualizations
# Numerical Summary 
summary(data)
num_data<-names(which(sapply(data,is.numeric)))# check the data types of variables
char_data<-names(which(sapply(data,is.character)))# check the number of levels of categorical variables

#Graphical summaries
hist(data$LotArea) 
hist(data$YearBuilt)
hist(data$OverallQual) 
hist(data$SalePrice) 
mean(data$SalePrice)
ggplot(data, aes(x = Alley)) + geom_bar() 
ggplot(data, aes(x = Utilities)) + geom_bar() 
ggplot(data, aes(x = LotConfig)) + geom_bar() 
ggplot(data, aes(x = Neighborhood)) + geom_bar() 
ggplot(data, aes(x = Condition1)) + geom_bar() 
ggplot(data, aes(x = Condition2)) + geom_bar()
ggplot(data, aes(x = OverallQual)) + geom_bar() 
ggplot(data, aes(x = OverallCond)) + geom_bar()
ggplot(data, aes(x = HouseStyle)) + geom_bar()

# storing data into data1
data1<-data
#Task 2: Divide houses based on their overall condition as follows:
#Poor if the overall condition is between 1 to 3.
#Average if the overall condition is between 4 and 6
#Good if the overall condition is between 7 and 10

data1$OverallCond <- factor(data1$OverallCond,
                            levels = 1:10,
                            labels = c('Poor', 'Poor', 'Poor',
                                       'Average', 'Average', 'Average',
                                       'Good', 'Good', 'Good', 'Good'))

str(data1)# looking at the structure of the data
data1 = data1[-c(1, 6, 45)] # removing the unnecessary variables

# separating the categorical and numerical variables 
num_vec_indx = c(1:2, 11, 13, 17, 23, 25:28, 33, 37, 40, 43:45)

# transformation of the numerical variables
# min-max and log transformation at a time addition with 5
log_t = function(y) log((y - min(y, na.rm = T)) / (max(y, na.rm = T) - min(y, na.rm = T)) + 5)

data1[num_vec_indx] =lapply(data1[num_vec_indx], log_t)
num_vec_indx = c(num_vec_indx, 48)

# making the categorical variables as factor

data1[-num_vec_indx] = as.data.frame(lapply(data1[-num_vec_indx], factor))

# to treat the missing value
library(missForest) 
data1 <- missForest(data1) # applying the missForest function to impute missing values 
data1$OOBerror # to see the imputation error

data1 = data1$ximp # to extract the data only

# checking the number of missing values 
sort(colSums(is.na(data1)))

# Fit the logistic regression model
model <- glm(OverallCond ~ ., data = data1, family = "binomial")
# View the summary of the model
summary(model)

# (b) Carry out a similar study using a different classification method to classify

install.packages('randomForest', dependencies = TRUE) 
library(randomForest) 

set.seed(13) # for setting the randomness fixed for all the runs 
classifier <- randomForest(OverallCond ~ ., data = data1, ntree = 1000)

# Display the random forest model and its goodness-of-fit measures
classifier

# Display a summary of the random forest model
summary(classifier)
y_pred = predict(classifier) # to predict by using the model 
importance(classifier) # to know the importance of the independent variables 

varImpPlot(classifier)

# Task 3:Predicting house prices:

# fitting multiple linear regression
regressor =lm(SalePrice ~ ., data = data1)
regressor # to show the regressor

# To test of hypothesis and see the summary of the model and to 
#know the goodness of fit of the model
summary(regressor)
# predicting the test set results
y_pred = predict(regressor, newdata = data1)
MSE_lm = mean(resid(regressor)^2) # Mean sum of squares of errors

# random forest
set.seed(137)
regressor = randomForest(SalePrice ~ ., data =data1, ntree = 500)
regressor
summary(regressor)
importance(regressor) # to know the importances of the independent variables
varImpPlot(regressor) # to plot the importances

# predicting the test set results
y_pred = predict(regressor, newdata = data1) # sum squares of errors
                 
MSE_rf = mean((data1$SalePrice - y_pred)^2)
# to compare mean sum of squares of errors from different models 
MSE = data.frame(MSE_lm, MSE_rf) 
MSE_sorted <- sort(MSE) # sort the MSE data frame by increasing order
MSE_sorted # print the sorted MSE data frame

# Task 3(b)

# train-test resampling for multiple regression model 
set.seed(137)
indx = sample(1:nrow(data1), round(0.8 * nrow(data1))) 
train_set = data1[indx,]
test_set = data1[-indx,]

# fitting multiple linear regression
regressor = lm(SalePrice ~ ., data = train_set)
regressor # to show the regressor                

# to test of hypothesis and see the summary of the model
summary(regressor)
# predicting the test set results
y_pred = predict(regressor, newdata = test_set)
MSE_lm_tt = mean((test_set$SalePrice - y_pred)^2) # sum squares of errors

# bootstrap resampling for multiple regression model 
MSE_lm_boot = 2^1000 
# setting a very large value
# loop for bootstrap
# Run a loop 5 times to perform bootstrapping and linear regression modeling
for (i in 1:5) {
  # Randomly select 90% of the rows in data2
  indx <- sample(1:nrow(data1), round(0.9 * nrow(data1)))
  # Create a new data frame with the selected rows
  data2 <- data1[indx,]
  # Fit a linear regression model to predict SalePrice using all other variables in data3
  regressor <- lm(SalePrice ~ ., data = data2)
  # Use the fitted model to predict SalePrice for the rows in data3
  y_pred <- predict(regressor, newdata = data2)
  # Calculate the mean squared error between the predicted and actual SalePrice in data3
  MSE_lm <- mean((data2$SalePrice - y_pred)^2)
  # Update the "best" linear regression model and its corresponding MSE if the current model has a lower MSE
  if (MSE_lm < MSE_lm_boot) {
    regressor_best <- regressor
    MSE_lm_boot <- MSE_lm
  }
}

# train-test resampling for random forest model 
set.seed(137)
indx = sample(1:nrow(data1), round(0.8 *nrow(data1)))
train_set = data1[indx,]
test_set = data1[-indx,]
# fitting multiple linear regression
regressor = randomForest(SalePrice ~ ., data = train_set, ntree = 500) 
regressor # to show the regressor
# to test of hypothesis and see the summary of the model
summary(regressor)
# predicting the test set results
y_pred = predict(regressor, newdata = test_set)
MSE_rf_tt = mean((test_set$SalePrice - y_pred)^2) # sum squares of errors
# bootstrap resampling for random forest model 
MSE_rf_boot = 2^1000 # setting a very large value 
# loop for bootstrap
# Run a loop 5 times to perform bootstrapping and random forest modeling
for (i in 1:5) {
  # Randomly select 90% of the rows in data2
  indx <- sample(1:nrow(data1), round(0.9 * nrow(data1)))
  # Create a new data frame with the selected rows
  data2 <- data1[indx,]
  # Fit a random forest model to predict SalePrice using all other variables in data3
  regressor <- randomForest(SalePrice ~ ., data = data2, ntree = 500)
  # Use the fitted model to predict SalePrice for the rows in data3
  y_pred <- predict(regressor, newdata = data2)
  # Calculate the mean squared error between the predicted and actual SalePrice in data3
  MSE_lm <- mean((data2$SalePrice - y_pred)^2)
  # Update the "best" random forest model and its corresponding MSE if the current model has a lower MSE
  if (MSE_lm < MSE_rf_boot) {
    regressor_best <- regressor
    MSE_rf_boot <- MSE_lm
  }
}

# to compare mean sum of squares of errors from different models
MSE = data.frame(MSE_lm, MSE_rf, MSE_lm_tt, MSE_rf_tt, MSE_lm_boot, MSE_rf_boot) 
sort(MSE) # to the model having lowest MSE

# Task 4
#we may consider to know the number of clusters between the variables OverallCond and OverallQual.
# taking the required data only
data2 = data.frame(data$OverallCond, data$OverallQual)

# using the dendrogram to find the optimal number of clusters 
dendrogram = hclust(d = dist(data2, method = 'euclidean'), method = 'ward.D') 
plot(dendrogram,main ='Dendrogram', xlab= 'Number of House',ylab = 'Euclidean distances')
# fitting hierarchical clustering to the data3
hc = hclust(d = dist(data2, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 8)
# visualizing the clusters
 install.packages('cluster',dep = T) 
 library(cluster)

 clusplot(data2, y_hc, lines = 0, shade = TRUE, color = TRUE, labels =
            2, plotchar = FALSE, span = TRUE, main = 'Clusters of Houses', xlab = 'OverallCond', ylab = 'OverallQual')










