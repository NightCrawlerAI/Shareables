#library(tidyverse)
library(readxl)
library(AER)
library(plm)
library(stargazer)
library(lmtest)
library(dplyr)
library(rpart)
library(rpart.plot)
library(RWeka)
library(quantmod)
library(e1071)
library(devtools)
library(broom)
library(caret)
library(partykit)
library(pls)
library(corrplot)
library(caTools)
library(olsrr)
library(ggplot2)
library(tseries)


adf.test(ns$EIA)


mean(national_storage$EIA)
sqrt(var(national_storage$EIA))

par(mfrow = c(2,2))
hist(national_storage$EIA)
hist(log(national_storage$EIA))

View(national_storage$EIA)

nseia <- mean(national_storage$EIA)

national_storage$EIA <- national_storage$EIA - nseia
View(national_storage$EIA)

#Read in the data
national_storage <- read.csv("P:/KJ/Coding Projects/Storage Regression (R Project)/national4.csv", header = TRUE, row.names = 'Date')#, colClasses = "character")

#Stationalized 
df_diff <- read.csv("P:/KJ/Coding Projects/Storage Regression (R Project)/df_differenced.csv", header = TRUE, row.names = 'Date')
hist(df_diff$EIA)
hist(log(df_diff$EIA))

sapply(national_storage, var, na.rm = TRUE) #Check for low variance
drops <- c("X123086", "X408920", "X146847", "X89088")
ns = national_storage[, !(names(national_storage) %in% drops)]
df_new <- sapply(ns, function(x) scale(x, scale = FALSE))
class(national_storage)
df_new.as_data.frame()
par(mfrow = c(2,1))
hist(national_storage$EIA)
hist(df_new$EIA)
View(df_new)
colMeans(df_new)
#, col = "orange", border = "blue", main = "Brent Distribution of Returns", xlab = "Last Price", xlim = c(40,200), breaks = seq(40,199, 1))

#Check the object type
#class(national_storage)
#View(national_storage)
#national_storage <- na.omit(national_storage)

#Create train and test sets
#dt = sort(sample(nrow(national_storage), nrow(national_storage)*.9)) #May be able to use a randomize sample function instead of static sampling
#train <- national_storage[dt,]
#test <- national_storage[-dt,]

#using caTools
?sample.split

sample <- sample.split(national_storage, SplitRatio = 0.9)
train2  <- subset(national_storage, sample == TRUE)
test2   <- subset(national_storage, sample == FALSE)


#df_train = train2[, -c("X123086", "X408920", "X146847")]#subset(train2, select = -c(X123086, X408920, X146847))
#df_test = test2[, -c("X123086", "X408920", "X146847")]
drops <- c("X123086", "X408920", "X146847", "X89088")

df_train = train2[, !(names(train2) %in% drops)]
df_test = test2[, !(names(test2) %in% drops)]


#round(cor(train2), 2)
#corrplot(cor(train2), order = "original", type = "full")
#################################


#high_correls <- national_storage[(cor(national_storage) > .9)]
#high_correls

#Remove outliers
#names_of_correls <- names(high_correls)
#high_cor_sub <- national_storage[names_of_correls,]

#skewless_train <- national_storage %>% anti_join(high_cor_sub)



#setup_cor <- findCorrelation(national_storage)
#setup_cor


#################################
national_unres <- lm(EIA ~ ., data = df_train)
summary(national_unres)

par(mfrow = c(2,2))
plot(national_unres)




#national_res <- lm(Total ~ ., data = df_train)
#summary(national_res)

#anova(national_unres, national_res) #Reject H0: Beta2 = Beta3 = 0

bptest(national_unres) #Fail to Reject H0: E(^var) = E(var)

#u2 <- (national_unres$residuals)^2

#national_gls <- lm(Total ~ ., data = df_train, weights = (1/u2))
#summary(national_gls)


#cor.table = cor(ns)
#corrplot(cor.table, type = 'lower', order = 'original')
#?corrplot
#cor(ns$Columbia.Gas.W, ns$DTI.W)



#plot(national_unres$residuals ~ national_unres$fitted.values, xlab = "Fitted Values", ylab = "Residuals")
#abline(h = 0, lty = "dashed")


#define intercept-only model
intercept_only <- lm(EIA ~ 1, data=df_train)


#define model with all predictors
#all2 <- lm(Total ~ . + I(NDD^2) - Honeoye.I - Leaf.River.N - TETCO.W - PGE.CGT.X - Boardwalk.X - Northern.Nat.W.3 - Questar.PL.I -
             #NGPL.W.2 - El.Paso.X - Date, data = national_storage)
#summary(all2)

all <- lm(EIA ~ ., data = df_train)
summary(all)

#anova(all, all2)
#cor(national_storage$NDD, national_storage$TETCO.X.1)


#perform stepwise regression
both <- step(intercept_only, direction='both', scope=formula(all), trace=0)
summary(both)

#Switch plot dimensions to 2x2
par(mfrow = c(2,2))
plot(both)
test <- predict(both, df_test)
summary(test)
summary(df_test$EIA)
MAE(df_test$EIA, test) # Stepwise Model (including outliers)


#################################################
#Identify outliers 
bothD <- cooks.distance(both)
outliers <- bothD[(bothD > (3 * mean(bothD, na.rm = TRUE)))]
outliers

#Remove outliers
names_of_outliers <- names(outliers)
outliers_subset <- df_train[names_of_outliers,]

#create the new df excluding the outliers 
skewless_df <- df_train %>% anti_join(outliers_subset)
class(skewless_df) #data.frame
length(skewless_df) #138

model2 <- lm(EIA ~ ., data = skewless_df) #skewless_df is not reading correctly in the ols model
summary(model2)

intercept_only2 <- lm(Total ~ 1, data=skewless_df)
model2_step <- ols_step_both_p(national_unres, progress = TRUE, details = TRUE)
model2_step$model
summary(both)
summary(national_unres)

??step

#view results of stepwise regression
both$anova

testing <- predict(all2, ns_test)
summary(testing)
summary(ns_test$Total)
testing2 <- predict(model2_step, df_test)
summary(testing2)
summary(df_test$EIA)

#bptest()
######################################################



#Partial Least Squares Regression 
#Using the kernelpls fit method and the leave-one-out validation method
  #Check accuracy by comparing to the classical orthogonal scores algorithm 
ns_plsr <- pcr(EIA ~ .,scale = TRUE, validation = 'CV', data = df_train, method = 'svdpc', ncomp = 17) 

summary(ns_plsr) #40 comps yield lowest MAE compared to other models 
plot(ns_plsr$residuals ~ ns_plsr$fitted.values)
plot(ns_plsr)
plot(df_test$EIA)
coefplot(ns_plsr)
abline(h = 0)
checker <- predict(ns_plsr, df_test)
summary(checker)
summary(df_test$EIA)
summary(test)
MAE(df_test$EIA, test)
MAE(df_test$EIA, checker)
R2(df_test$EIA, test, form = "traditional")
fitted(ns_plsr)
pcr_coefs
model.frame(ns_plsr)
model.matrix(ns_plsr)
R2(ns_plsr)
MSEP(ns_plsr)
mvrValstats(ns_plsr, "all")


predplot(ns_plsr)
coefplot(ns_plsr)
pcr_coefs = coef(ns_plsr)
dim(checker)
checker



ggplot(df_test, aes(x=predict(ns_plsr, EIA), y=EIA)) + 
  geom_point() +
  geom_abline(intercept=0, slope=1) +
  labs(x='Predicted Values', y='Actual Values', title='Predicted vs. Actual Values')

class(ns_plsr)
?mvr
?ggplot2


rlang::last_error()#where did the error occur
rlang::last_trace()


length(df_test$EIA)
summary(df_test$EIA)
MAE(df_test$EIA, checker) #PCR model w/ ncomp=40 produces the lowest MAE @ 9.191816
# Plot the R2
validationplot(ns_plsr, val.type = "MSEP")
pcr_r2 = R2(df_test$EIA, checker)
class(ns_plsr)
?plsr
#As a result of hyper-parameter tuning, the chosen PLSR model will be the PCR model using cross validation instead of leave-one-out, MAE=9.489

#Comparison Plots
plot(national_gls)
plot(national_unres)
plot(national_res)
summary(gls)
summary(gls_res)
summary(national_unres)
#summary(ns_plsr)
#ns_plsr$coefficients

#Create the xts zoo 
national_storage$Date <-as.factor(national_storage$Date)
national_storage$Date <-strptime(national_storage$Date,format="%m/%d/%Y") #defining what is the original format of date
national_storage$Date <-as.Date(national_storage$Date,format="%Y-%m-%d")
ns <- xts(national_storage[,-c(1)],order.by = as.Date(national_storage[,1]))

#Create train and test sets
dt = sort(sample(nrow(ns), nrow(ns)*.8)) #May be able to use a randomize sample function instead of static sampling
trainset <-  window(ns, end = "2021-07-30")
testset <- window(ns, start = "2021-08-06")
class(testset)
df_train <- data.frame(trainset)
df_test <- data.frame(testset)
class(df_train)
??svm

#Train SVR models (using eps-regression)
#svmt <- svm(Total ~ ., data = df_train, kernel = "sigmoid", cost = 1, epsilon = 0.1)
#svmt

#svmt2 <- svm(Total ~ ., data = df_train, kernel = "polynomial")
#svmt2


#Tuning of Hyperparameters
svmt3<- svm(Total ~ ., data = df_train, kernel = "radial basis", cost = 1, epsilon = 0.1)
svmt3

#Compare the SVR models
svmp <- predict(svmt, df_test$Total)
svmp

svmp2 <- predict(svmt2, df_test$Total)
svmp2

svmp3 <- predict(svmt3, df_test$Total)
svmp3
df_test

#table(svmp, df_test$Total)

#Actuals
summary(df_test$Total)
#Sigmoid
summary(svmp)
#Polynomial
summary(svmp2)
#Linear
summary(svmp3)

plot(df_test$Total, pch=16)

points(df_test$Total, svmp3, col = "red", pch = 4)
points(df_test$Total, svmp2, col = "blue", pch = 2)
points(df_test$Total, svmp, col = "orange", pch = 3)
points(df_test$Total, p, col = "purple", pch = 5)
points(df_test$Total, p2, col = "yellow", pch = 6)
points(df_test$Total, both_p, col = "green", pch = 0)

lin_r2 = R2(df_test$Total, svmp3, form = "traditional")
poly_r2 = R2(df_test$Total, svmp2, form = "traditional")
sig_r2 = R2(df_test$Total, svmp, form = "traditional")
m5p_r2 = R2(df_test$Total, p2, form = "traditional")
both_r2 = R2(df_test$Total, both_p, form = "traditional")

#Linear SVM
lin_r2
MAE(df_test$Total, svmp3)

#Polynomial SVM
poly_r2
MAE(df_test$Total, svmp2)

#Sigmoid SVM
sig_r2
MAE(df_test$Total, svmp)

#M5' Algorithm 
m5p_r2
MAE(df_test$Total, p2)

#Stepwise Regression
both_r2
MAE(df_test$Total, both_p)



#View(national_storage[1,-c(1)])
#as.Date(national_storage$Date)



#Introduce unseen data
#unseen <- read.table("P:/KJ/Coding Projects/Storage Regression (R Project)/national unseen data.csv", header = TRUE, sep =",")
#colnames(unseen)
#unseen = subset(unseen, select = -c(I.NDD.2.))
#colnames(unseen)
#By slicing, remove Date column
#df = subset(national_storage, select = -c(Date))




#Construct a recursive partitioning model 
m <- rpart(Total ~ ., data = df_train)
summary(m)

#ber <- rpart(Total ~ ., data = df)
#summary(ber)

#Check model constraints and fine tune the methodology
#?rpart.control

#Plot the results at each node of the tree
rpart.plot(m, digits =3)

#Predict actuals (unseen) from the M5P reression tree model 
#pkg_pred <- rpart.predict(ber, unseen, type = "vector")
#Predicted Values via M5P
#pkg_pred
#plot(national_storage$Total)
#abline(h=0, lty = "dashed")

#Actual values
#unseen$Total
#Print the rules of the model to confirm methodology
#rpart.rules(ber)


p <- predict(m, df_test)
summary(p)
summary(df_test$Total)
#p
#summary(test$Total)
#cor(p, test$Total)

#Create Mean Absolute Error function
MAE<- function(actual, predicted) {
  mean(abs(actual-predicted))
}


MAE(df_test$Total, p)


m2 <- M5P(Total ~ ., data = df_train)
summary(m2)

#To plot the M5P model 
plot(m2)
#summary(ber)

p2 <- predict(m2, df_test)
summary(p2)
summary(df_test$Total)

p2
plot(p2)
abline(h=0, lty="dashed")



#unseen$Total

#cor(p2, unseen$Total)

MAE(df_test$Total, p2)


#library(gmodels)
#CrossTable(both$residuals, both$fitted.values)

#train
#class(train)

#x_train <- subset(train, select = -c(Total))
#x_train

#length(x_train)
#length(y_train)
#length(test)

#y_train <- subset(train, select = c(Total))
#y_train

#class(y_train)

#dataset = cbind(y_train, x_train)
#class(dataset)
#dataset

#x_test <- subset(test, select = -c(Total))
#x_test

#y_test <- subset(test, select = c(Total))
#y_test

#ols <- lm(Total ~ . - Date, data = df_train)
#summary(ols)

#prediction = predict(ols, df_test)

#col_heads <- list(names(national_storage))
#head_names <- colnames(national_storage)
#write.csv(head_names,"national_colnames.csv")


count(df_test)
View(df_test)

ns_test = head(national_storage, n = 53)
View(ns_test)

ns_r2 = R2(df_test$Total, testing, form = "traditional")

#Linear SVM
ns_r2
MAE(ns_test$Total, all2)
all2$fitted.values

#View final model
both$coefficients
MAE(ns_test$Total, all2)

ns_test <- subset(ns_test, select = -c(Date))
colnames(ns_test)


plot(both$residuals ~ both$fitted.values, xlab = "Fitted Values (Both)", ylab = "Residuals (Both)")
abline(h = 0, lty = "dashed")
both_p <- predict(both, df_test)
both_p



#sink("national_coefs.csv")
#summary(both)
#sink()

#Check where it saved
getwd()

#Save the file
tidy_both <- tidy(both)
write.csv(pcr_coefs,"national_coefs_final.csv")

#### SVM Model

# By default R comes with few datasets. 
#data = national_storage
#data = data[, -c(1)]
#dim(data)  # 32 11
#View(data)



#Sample Indexes
#indexes = sample(1:nrow(data), size=0.2*nrow(data))

# Split data
#test = data[indexes,]
#dim(test)  # 6 11



#train = data[-indexes,]
#dim(train) # 26 11



#library(kernlab)
#class(test)
#View(test)

#m < ksvm(x = , y = , data = train, kernel = "rbfdot", )
#m 
#?ksvm

#install.packages("e1071")
#summary(m)
#p <- predict(m, test, type = "response")
#summary(p)