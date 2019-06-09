#############################################################################################
# CAPSTONE PROJECT
# Project Submission : Choose Your Own Project 

###################### INDIAN LIVER PATIENT RECORDS ##################################

# This dataset is downloaded from kaggle website. 
# https://www.kaggle.com/uciml/indian-liver-patient-records
# The dataset contains 416 liver patient records and 167 non liver patient records. 
# The dataset has 10 medical parameters of the patients which can be used in modeling the 
# machine learning algorithm for predicting the chances of liver disease. The 11th variable "Dataset" 
# is the class label which divides the patients to liver diseased or not.

# Start with loading the libraries
library(ggplot2)
library(corrplot)
library(randomForest)
library(caret)

# Load the data
# data <- read.csv("indian_liver_patient.csv", header=TRUE, 
#                  na.strings = c("NA","na",""," ","?"), stringsAsFactors = FALSE)

# Load the data
load("rdas/data.rda")

# Aim here is to:
# Explore the dataset and the variables, 
# Clean the data, 
# Prepare the variable for modeling

head(data)
str(data)
# Can appreciate the data dimensions, variables and the class

# Explore the label column
table(data$Dataset)

# Missing values
sapply(data, function(x) sum(is.na(x)))
# Only 4 missing values

# EDA and cleaning of variables
# EDA helps to get insight to the spread of variables

# Age
table(data$Age)
# Patient age spread across 4-90 yrs

ggplot(data, aes(data$Age,fill=factor(data$Dataset))) + geom_bar(position = "dodge")

ggplot(data, aes(data$Age, data$Dataset, color=factor(data$Dataset))) + 
  geom_point(position = "jitter")
# The points are given different colors to different label in Dataset variable. "jitter" helps 
# spread the points and help appreciated the spread.
# This can be grouped in to few age categories. Levels are reduced to 7 categories here.

data$Age <- replace(data$Age, data$Age <=12, "1")
data$Age <- replace(data$Age, data$Age >12 & data$Age <=18, "2")
data$Age <- replace(data$Age, data$Age >18 & data$Age <=40, "3")
data$Age <- replace(data$Age, data$Age >40 & data$Age <=55, "4")
data$Age <- replace(data$Age, data$Age >55 & data$Age <=65, "5")
data$Age <- replace(data$Age, data$Age >65 & data$Age <=75, "6")
data$Age <- replace(data$Age, data$Age >75, "7")
table(data$Age)

# Type of variable is changed to integer for modeling
class(data$Age)
data$Age <- as.integer(data$Age)

# Cleaning for variable Gender
table(data$Gender)
ggplot(data, aes(data$Gender,fill=factor(data$Dataset))) + geom_bar(position = "dodge")
# Males have higher diseased proportion

# Converting the variable to integer
data$Gender <- replace(data$Gender, data$Gender == "Male", "1")
data$Gender <- replace(data$Gender, data$Gender == "Female", "2")
data$Gender <- as.integer(data$Gender)

# Exploring the variable Total_Bilirubin

ggplot(data, aes(data$Total_Bilirubin, data$Dataset, color=factor(data$Dataset))) + 
  geom_point(position = "jitter")
# Can aprreciate the spread of values.
# Normal Total bilirubin value ranges between : 0.1 to 1.2 mg/dL (1.71 to 20.5 µmol/L)
# High increase in values can be seen, which even reaches 60mg/dl and above in diseased group

# The mean value also is very high in diseased group
tapply(data$Total_Bilirubin, data$Dataset, mean, na.rm=TRUE)

# Exploring the variable Direct_Bilirubin
# Appreciate the spread and mean values in two groups
# Normal level of Direct bilirubin: less than 0.3 mg/dL (less than 5.1 µmol/L)
ggplot(data, aes(data$Direct_Bilirubin, data$Dataset, color=factor(data$Dataset))) + 
  geom_point(position = "jitter")
tapply(data$Direct_Bilirubin, data$Dataset, mean, na.rm=TRUE)
# High mean value of 1.9 is seen in diseased grouped when the normal value is less than 0.3mg/dl

# Explore variable Alkaline_Phosphotase
ggplot(data, aes(data$Alkaline_Phosphotase, data$Dataset, color=factor(data$Dataset))) + 
  geom_point(position = "jitter")
tapply(data$Alkaline_Phosphotase, data$Dataset, mean, na.rm=TRUE)
# Normal range for serum ALP level is 20 to 140 IU/L

# Explore the variable Alamine_Aminotransferase
ggplot(data, aes(data$Alamine_Aminotransferase, data$Dataset, color=factor(data$Dataset))) + 
  geom_point(position = "jitter")
tapply(data$Alamine_Aminotransferase, data$Dataset, mean, na.rm=TRUE)
# Normal values for ALT between 7 to 56 units per liter

# Explore the variable Aspartate_Aminotransferase
ggplot(data, aes(data$Aspartate_Aminotransferase, data$Dataset, color=factor(data$Dataset))) + 
  geom_point(position = "jitter")
tapply(data$Aspartate_Aminotransferase, data$Dataset, mean, na.rm=TRUE)
# Normal AST is reported between 10 to 40 units per liter

# Explore the variable Total_Protiens
ggplot(data, aes(data$Total_Protiens, data$Dataset, color=factor(data$Dataset))) + 
  geom_point(position = "jitter")
tapply(data$Total_Protiens, data$Dataset, mean, na.rm=TRUE)
# Normal range for total protein is between 6 and 8.3 grams per deciliter (g/dL).

# Explore the variable Albumin
ggplot(data, aes(data$Albumin, data$Dataset, color=factor(data$Dataset))) + 
  geom_point(position = "jitter")
tapply(data$Albumin, data$Dataset, mean, na.rm=TRUE)
# The normal range of Albumin is 3.5 to 5.5 g/dL or 35-55 g/liter.

# Explore the variable Albumin_and_Globulin_Ratio
ggplot(data, aes(data$Albumin_and_Globulin_Ratio, data$Dataset, color=factor(data$Dataset))) + 
  geom_point(position = "jitter")
tapply(data$Albumin_and_Globulin_Ratio, data$Dataset, mean, na.rm=TRUE)
# The normal A/G ratio is 0.8-2.0.

# The variation in medical parameters could be the result of any medical condition or illness, 
# from which the condition of liver disease needs to be differentiated.

# Remove the observation with missing value. The 4 observation being very less in number are removed.
data<-na.omit(data)

# Dimention of the dataset
dim(data)

# Find the correlation between variables in the dataset
data_cor <- round(cor(data, use="pairwise.complete.obs"), 2)
corrplot(data_cor)
# Few of thoses varibales are correlated

# Label column is converted to factor variable before modeling
data$Dataset <- as.factor(data$Dataset)

# Set the seed to make the results repeateable
set.seed(123)

# The dataset is divide to 75% train and 25% test observations.
# The train set is used to train the model and test set is used to validate the model performance.

indices <- sample(2, nrow(data), replace = T, prob = c(0.75, 0.25))
train <- data[indices == 1, ]
test <- data[indices == 2, ]

# Modeling
# Random forest algorithm is used
set.seed(123)
rf.model <- randomForest(Dataset ~ ., data = train, do.trace = T)

# Prediction
rf.predict <- predict(rf.model, test[,-11], type = "class")

# Confusion matrix to get the idea of prediction against true labels
confusionMatrix(rf.predict, test$Dataset)
# Accuracy: 0.6809
# Sensitivity: 0.9000
# Specificity: 0.2941

# Let try to improve the model by improving the specificity which is very low in this model.
# It is important to avoid false positives.
# So need to tune the model.

# Here the probability of falling to the 2 labelled groups are produced.
# We will try to find which cutoff probability will separate the cases to the 2 groups by giving 
# importance to both the sensitivity and specificity.
# Currently the cutoff is set to 0.50 and with the help of plot a minimum value to the difference 
# of sensitivity and specificity is calculated.
# Hopefully we expect this cutoff probability will help produce a balanced sensitivity and 
# specificity and good overall accuracy.

# Get probability
rf_predict <- data.frame(predict(rf.model, test[,-11], type = "prob"))
summary(rf_predict$X1)
summary(rf_predict$X2)

# Get Predicition with 0.05 cutoff
predicted <- factor(ifelse(rf_predict$X1 >= 0.5, "1", "2"))

# Create Function to produce confusion matrix values for a series of probabilities
perform_fn <- function(cutoff) 
{
  predicted <- factor(ifelse(rf_predict$X1 >= cutoff, "1", "2"))
  conf <- confusionMatrix(predicted, test$Dataset, positive = "1")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  out <- t(as.matrix(c(sens, spec, acc))) 
  colnames(out) <- c("sensitivity", "specificity", "accuracy")
  return(out)
}


# Perform confusion matrix values for all values in 's'
s = seq(.01,.95,length=100)
OUT = matrix(0,100,3)
for(i in 1:100){
  OUT[i,] = perform_fn(s[i])
}

# Plot the confusion matrix cutoff with accuracy, sensitivity and specificity values
plot(s, OUT[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",
     lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT[,2],col="darkgreen",lwd=2)
lines(s,OUT[,3],col=4,lwd=2)
box()
legend("topright",col=c(2,"darkgreen",4),lwd=c(1,1,1),
       c("Sensitivity","Specificity","Accuracy"), cex=0.4)

# Pull the Cutoff value
cutoff <- s[which.min(abs(OUT[,1]-OUT[,2]))]
cutoff
# 0.65565 seems to be good grouping probability value which can separate the patients in to diseased 
# and non-diseased, with a balanced weightage to sensitivity & specificity and accuracy.
 
# We hope to give good prediction on the disease status of a patient with this random forest algorithm
# by deciding on to be diseased, if the probability of prediction is above 0.655 and not diseased below
# the probability of 0.655

# Let see the values with the probability of 0.655, the cutoff value.
test_cutoff <- factor(ifelse(rf_predict$X1 >= cutoff, "1", "2"))

conf_final <- confusionMatrix(test_cutoff, test$Dataset, positive = "1")

acc <- conf_final$overall[1]
sens <- conf_final$byClass[1]
spec <- conf_final$byClass[2]

acc # 0.6950355 
sens # 0.7
spec #  0.6862745 

# This seems to give a almost 70% accuracy in prediction of the patient status (diseased or not).
# Means we could predict the patient status with 70% accuracy.

# Random forest gives the opportunity to find the importance of variable in prediction.
# Let us plot the variable importance
# Gini index or the amount of disorder in the dataset when decreased to maximum, will produce a more
# homogenous nodes, or the datapoints in the resulting nodes will be more similar. So the feature or 
# the variable which helps to give a better reduction in gini will have more importance in classifying the observations in the
# dataset

varImpPlot(rf.model)
# Alkaline Phosphatase, Aspartate Aminotransferase and Alamine Aminotransferase are 3 most important 
# variables in classfying the dataset. We already saw that the values of these variables in the
# 2 groups varied widly.

