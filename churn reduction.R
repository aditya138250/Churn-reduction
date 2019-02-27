rm(list = ls())

setwd('E:/Project/Churn reduction/R')

getwd()

library("dplyr")
library("ggplot2")
library("data.table")
library('scales')
library('psych')
library('corrgram')
library('ggcorrplot')
library('rpart')
library('randomForest')
library('caret')
library('MASS')
library('e1071')
library('gbm')
library('DMwR')
library('mlr')
library('dummies')
library('DataCombine')
library('C50')

#loading dataset

df = read.csv('E:/Project/Churn reduction/R/Train_data.csv',header = T)
df1 = read.csv('E:/Project/Churn reduction/R/Test_data.csv',header = T)

#Binding two dataset in one
churn_red =  data.frame(rbind(df, df1))

#Cleaning environment
rmExcept("churn_red")

#Checking if any NA are present
sum(is.na(churn_red))    ##We can clearly see that there are 0 missing values in given dataset.

colnames(churn_red)

str(churn_red)


#Since we do not require few variables because we have to predict the churn score based on usage pattern so 
# variables like "area code", "phone number" and "state" are not important so we will drop them.

churn_red = subset(churn_red, select = -c(state,area.code,phone.number))



#Storing continuous and categorical variable in different objects
variable_num = c('account.length', 'number.vmail.messages', 'total.day.minutes', 'total.day.calls', 'total.day.charge',
                'total.eve.minutes', 'total.eve.calls', 'total.eve.charge','total.night.minutes','total.night.calls',
                'total.night.charge','total.intl.minutes', 'total.intl.calls','total.intl.charge',
                'number.customer.service.calls')

variable_cat = c('international.plan', 'voice.mail.plan', 'Churn')

##Data Manupulation; convert string categories into factor numeric
# No=1 and yes = 2
for(i in 1:ncol(churn_red)){
  
  if(class(churn_red[,i]) == 'factor'){
    
    churn_red[,i] = factor(churn_red[,i], labels=(1:length(levels(factor(churn_red[,i])))))
    
  }
}


#################  Outlier Analysis   #####################
# Boxplot for continuous variables
for (i in 1:length(variable_num))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (variable_num[i]), x = "Churn"), data = subset(churn_red))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "orange" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=variable_num[i],x="Churn")+
           ggtitle(paste("Box plot of churn for",variable_num[i])))
}

# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)
gridExtra::grid.arrange(gn7,gn8,gn9,ncol=3)
gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)
gridExtra::grid.arrange(gn13,gn14,gn15,ncol=3)


#Replace all outliers with NA and impute with knn
##create NA on "account.length"

for(i in variable_num)
{
  val = churn_red[,i][churn_red[,i] %in% boxplot.stats(churn_red[,i])$out]
  #print(length(val))
  churn_red[,i][churn_red[,i] %in% val] = NA
}

sum(is.na(churn_red))

# Imputing missing values
churn_red = knnImputation(churn_red,k=3)

sum(is.na(churn_red))

#Boxplot after outlier removal and replacement
for (i in 1:length(variable_num))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (variable_num[i]), x = "Churn"), data = subset(churn_red))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "orange" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=variable_num[i],x="Churn")+
           ggtitle(paste("Box plot of Churn for",variable_num[i])))
}

# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)
gridExtra::grid.arrange(gn7,gn8,gn9,ncol=3)
gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)
gridExtra::grid.arrange(gn13,gn14,gn15,ncol=3)


################### Feature Selection   ########################
## Correlation Plot 
ggcorrplot(cor(churn_red[,variable_num]),method = 'square', lab = T, title =  'Correlation plot',
           ggtheme = theme_dark())


## Chi-squared Test of Independence
factor_index = sapply(churn_red,is.factor)
factor_data = churn_red[,factor_index]

for (i in variable_cat)
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}



## Dimension Reduction
churn_red = subset(churn_red,select = -c(total.day.minutes,total.night.minutes,total.eve.minutes,
                                         total.intl.minutes))


#Since we have dropped few variables and stored in churn_red. So here we have to updated the variable
#and store them in new object.
variable_num_update = c('account.length', 'number.vmail.messages','total.day.calls', 'total.day.charge',
                        'total.eve.calls', 'total.eve.charge','total.night.calls',
                        'total.night.charge','total.intl.calls','total.intl.charge',
                        'number.customer.service.calls')

variable_cat_update = variable_cat = c('international.plan', 'voice.mail.plan', 'Churn')


##################################Feature Scaling################################################
#Normality check
for (i in variable_num_update) {
  d = density(churn_red[,i])
  plot(d, type="n", main=variable_num_update[i])
  polygon(d, col="red", border="gray")
  
}

#Since data is normally distributed we will standardise data for further analysis

# #Standardisation
for(i in variable_num_update){
  print(i)
  churn_red[,i] = (churn_red[,i] - mean(churn_red[,i]))/sd(churn_red[,i])
}




###### Checking VIF to see if there is multicollinearity
#install.packages('usdm')
library(usdm)
vif(churn_red[,variable_num_update])  ## It will calculate vif

vifcor(churn_red[,variable_num_update], th = 0.9)


###################################Model Development#######################################
#Clean the environment
rmExcept('churn_red')

#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(churn_red$Churn, p = .80, list = FALSE)
train = churn_red[ train.index,]
test  = churn_red[-train.index,]

##Decision tree for classification
#Develop Model on training data
C50_model = C5.0(Churn ~., train, trials = 100, rules = TRUE)

#Summary of DT model
summary(C50_model)

#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")

#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-14], type = "class")

##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$Churn, C50_Predictions)
confusionMatrix(ConfMatrix_C50)

#Recall rate(True positivve)
(TP*100)/(TP+FN)
(89*100)/(89+52)


# Accuracy : 93.99%
#Recall rate : 63.12 


###Random Forest
RF_model = randomForest(Churn ~ ., train, importance = TRUE, ntree = 500)

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-14])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$Churn, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

#Recall rate(True positivve)
(TP*100)/(TP+FN)
(78*100)/(78+63)

# Accuracy : 93.29%
#Recall rate : 55.31% 


#Logistic Regression
logit_model = glm(Churn ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")


#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)

logit_Predictions


##Evaluate the performance of classification model
ConfMatrix_RF = table(test$Churn, logit_Predictions)

ConfMatrix_RF

#Recall rate(True positivve)
(TP*100)/(TP+FN)
(12*100)/(12+129)

# Accuracy
841+12
853/999

# Accuracy : 85.38%
#Recall rate : 8.5% 
