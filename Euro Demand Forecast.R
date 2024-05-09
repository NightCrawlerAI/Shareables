library(tidyverse)
library(readxl)
library(AER)
library(plm)
library(stargazer)
library(lmtest)


data <- read.csv("C:/Users/kjones/Downloads/European Demand Forecast Inputs.csv")
#View(data)

# Remove missing values
data <- na.omit(data)

y1 <- data$LDZ
y2 <- data$Gas_Power
y3 <- data$Industrial

x1 <- data$Weekday
x2 <- data$Temps
x3 <- x1^3
x4 <- x2^3
x5 <- x1^4
x6 <- x2^4
x7 <- x1^5
x8 <- x2^5
x9 <- x1*x2
x10 <- x3*x4
x11 <- x5*x6
x12 <- x7*x8

data_final <- data.frame(y1, y2, y3, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)

# Fit the model
#fit <- lm(Industrial ~ poly(Temps, degree=1) * poly(Weekday, degree = 1), data=data) 

LDZ_ols <- lm(y1 ~ . -y2 -y3, data = data_final)
summary(LDZ_ols)

GTP_ols <- lm(y2 ~ . -y1 -y3, data = data_final)
summary(GTP_ols)

IND_ols <- lm(y3 ~ . -y1 -y2, data = data_final)
summary(IND_ols)



# Summary of the model
summary(fit)
bptest(fit)


u2 <- (fit$residuals)^2 
gls <- lm(Industrial ~ Temps  * Weekday, weights=(1/u2), data = data)
summary(gls)


fit2 <- lm(Industrial ~ poly(Temps, degree = 1) * poly(Weekday, degree = 1), data=data) 

summary(fit2)  # get detailed info


fit3 <- lm(Gas_Power ~ poly(Temps, degree = 1) * poly(Weekday, degree = 1), data=data) 

summary(fit3)  # get detailed info


#predict values using the fitted model:
#predicted_values <- predict(fit, newdata=data.frame(x=new_x_values))
#######################################################

library(randomForest)

# read the data
data <- read.csv("C:/Users/kjones/Downloads/European Demand Forecast Inputs.csv")

# Remove missing values
data <- na.omit(data)

# fit model
fit <- randomForest(LDZ ~ . - Date - Gas_Power - Industrial, data=data)

# print the summary
summary(fit)

#########################################

install.packages('boot')
library(boot)
set.seed(1)

# read the data
data <- read.csv("C:/Users/kjones/Downloads/European Demand Forecast Inputs.csv")

# Remove missing values
data <- na.omit(data)

y <- data$LDZ
x1 <- data$Weekday
x2 <- data$Temps
x3 <- x1^3
x4 <- x2^3
x5 <- x1^4
x6 <- x2^4
x7 <- x1^5
x8 <- x2^5
x9 <- x1*x2
x10 <- x3*x4
x11 <- x5*x6
x12 <- x7*x8

data <- data.frame(y, x1, x2)

# Set maximum degree
max_degree <- 5

# Initialize vector to store CV errors
cv_errors <- rep(0, max_degree)

# Loop over degrees 1 through max_degree
for (degree in 1:max_degree) {
  
  # Fit model and calculate CV error for each degree
  print(degree)
  glm_fit <- glm(y ~ poly(x1,degree) + poly(x2,degree), data=data)
  print(summary(glm_fit))
  
  cv_errors[degree] <- cv.glm(data, glm_fit, K=10)$delta[2]
}

# Find degree with smallest CV error
best_degree <- which.min(cv_errors)
best_degree

#summary(glm_fit)
