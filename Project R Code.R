#Final Project - ME8135
#Amir Ghaderi 500794236

#Install Packages
install.packages("class")
install.packages("e1071")
install.packages("ipred")
install.packages("randomForest")
install.packages("C50")
install.packages("tree")
install.packages('neuralnet')
install.packages("corrplot")
install.package("ggplot2")


#Load Packages 
library(class)
library(e1071)
library(ipred)
library(randomForest)
library(C50)
library(tree)
library(neuralnet)
library(corrplot)
library(ggplot2)

#Import Data
mydata <- read.csv("UCI_Credit_Card.csv")
colnames(mydata)[25] <-"class"
mydata <- mydata[,c(2,3,4,5,6,13,14,15,16,17,18,25)]
mydata2 <- mydata
mydata <- mydata[order(mydata$class),]
mydata <- mydata[16729:30000,]
data_train$class <- as.factor(data_train$class)
data_test$class <- as.factor(data_test$class)
table(mydata$class)

#Feature Normalization 
normalize <- function(x) {
  return( (x - min(x))/ (max(x)- min(x)))
}

mydata_class <-mydata$class
mydata <- as.data.frame(lapply(mydata[,c(1:11)], normalize))
mydata$class <- mydata_class

#Randomize the data
set.seed(1234)
gp <- runif(nrow(mydata))  
mydata <- mydata[order(gp),]

#Data Splitting
data_train <-mydata[1:10617,]
data_test <- mydata[10618:13272,]


#KNN
model_knn <- knn(train = data_train[,1:11], test = data_test[,1:11], cl = data_train[,12], k = 115)
table(data_test[,12], model_knn)
mean(model_knn == data_test[,12])

#SVM
#Cross Validation
tuned <- tune(svm, class~.,data=data_train, kernel="linear", ranges=list(cost=c(0.001,0.01,0.1,1,10,100)))
summary(tuned)

#model Training
model_svm <- svm(class~., data=data_train, kernel = "linear", cost=10, scale = FALSE, type = "C-classification")
print(model_svm)
p <- predict(model_svm, data_test[,1:11], type = "class")
table(data_test[,12],p)
mean(p == data_test[,12])

#Naive Bayes
data_train <- as.data.frame(sapply(data_train,as.factor))
data_test <- as.data.frame(sapply(data_test,as.factor))

model_NB <- naiveBayes(class~., data=data_train)
p <- predict(model_NB, data_test[,1:11], type = "class")
table(data_test[,12],p)
mean(p == data_test[,12])

#Decision Trees
#Tree
model_tree <- tree(class~.,data_train)
p <- predict(model_tree,data_test[,1:11], type="class")
table(data_test[,12],p)
mean(p == data_test[,12])

#Tree Pruning
#cv_tree <- cv.tree(model_tree, FUN=prune.misclass)
#names(cv_tree)
#plot(cv_tree$size,cv_tree$dev,type ="b")
#pruned_model=prune.misclass(model_tree, best=2)
#tree_pred = predict(pruned_model,data_test[,1:11],type="class")
#table(data_test[,12],p)
#ean(p == data_test[,12])

#Bagging
fit <- bagging(class~., data=data_train,type="class")
p <- predict(fit, data_test[,1:11], type="class")
table(data_test[,12],p)
mean(p == data_test[,12])

# Random Forest
fit <- randomForest(class~., data=data_train)
p <- predict(fit, data_test[,1:11], type="class")
table(data_test[,12],p)
mean(p == data_test[,12])

#C5.0
fit <- C5.0(class~., data=data_train, trials=10)
p <- predict(fit, data_test[,1:11], type="class")
table(data_test[,12],p)
mean(p == data_test[,12])

#Neural Networks

str(data_train)
nnp <- neuralnet(class~SEX+AGE+LIMIT_BAL, data = data_train, hidden=40,lifesign = "minimal",linear.output = FALSE,threshold = 0.1)

temp_test <- subset(data_test, select = c("SEX","AGE","LIMIT_BAL"))
creditnet.results <- compute(nnp,temp_test)
table(data_test[,12],round(creditnet.results$net.result))
mean(round(creditnet.results$net.result) == data_test[,12])

#Export Data
write.csv(mydata, file="mydata.csv")










