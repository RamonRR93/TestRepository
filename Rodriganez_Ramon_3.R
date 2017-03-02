#
#
# Business Analytics
# Assignment 3
# Ramon Rodriganez - rr3088
#
#

setwd('~/Github/TestRepository')



#///////////////// PROBLEM 1 ///////////////////////

#a############

Orange_Data = read.csv("OrangeJuice.csv")

#We check how does the data look like
head(Orange_Data)

#Column "Purchase" is binary: CH or MM. Column Store is qualitative: ID goes from 1 to 4 and 7.

#We start adding binary variables for the different stores
Orange_Data$Store1 <- ifelse(Orange_Data$StoreID == 1, 1, 0)
Orange_Data$Store2 <- ifelse(Orange_Data$StoreID == 2, 1, 0)
Orange_Data$Store3 <- ifelse(Orange_Data$StoreID == 3, 1, 0)
Orange_Data$Store4 <- ifelse(Orange_Data$StoreID == 4, 1, 0)

#We clean the StoreID Column. We also do not need column "X"
Orange_Data = subset(Orange_Data, select = - c(StoreID,X,SalePriceMM,SalePriceCH,PriceDiff))

#Now we determine MM as a buy and CH as not a buy:
Orange_Data$Purchase = ifelse(Orange_Data$Purchase == "MM",1,0)

#We assign the data to three different groups:
set.seed(1337)
RowAssignment = sample(c("train","validate","test"), nrow(Orange_Data), rep = TRUE)
train_rows = ifelse(RowAssignment == "train",TRUE,FALSE)
validate_rows = ifelse(RowAssignment == "validate",TRUE,FALSE)
test_rows = ifelse(RowAssignment == "test",TRUE,FALSE)

summary(Orange_Data[train_rows,])



#b############ - LOGISTIC REG

model1.fit=glm(Purchase~.,data=Orange_Data[train_rows,],family =binomial)

# Summary & coefficients are
summary (model1.fit)
coef(model1.fit)
summary(model1.fit)$coef



#c########### - LASSO

#install.packages("glmnet")
library(glmnet) 
#We use 5-fold CV for lambda=10^-3,10^-2,10^-1,1,10^1,10^2,10^3
grid=10^(-3:3) #set sequence of lambdas we want to test

#prepare the arguments for glmnet()
x=model.matrix(Purchase~.,Orange_Data)[,-1]
y=as.factor(Orange_Data$Purchase)

#Find optimum lambda
cv.out=cv.glmnet(x[train_rows,],y[train_rows],alpha=1,lambda=grid,family="binomial",nfolds=10)
bestlam=cv.out$lambda.min

#Train model with best value of lambda on the training set
model2.fit=glmnet(x[train_rows,], y[train_rows], alpha=1, lambda=bestlam, family="binomial")

# Summary & coefficients
coef(model2.fit)




#d######### - TREE

#install.packages("tree")
library(tree)
tree.mod = tree(Purchase~., data=Orange_Data[train_rows,])
# We use k-fold cross validation to choose the best # of leaves, k = 10
cv.out.tree = cv.tree(tree.mod, K = 10)  

plot(cv.out.tree$size, cv.out.tree$dev, type='b')
#optSize = cv.out.tree$size[which.min(cv.out.tree$dev)]
#Here we can use the code above. However, for simplicity decide to take a low value 
#which already gives pretty good results and makes the tree much simpler. In this case
optSize = 3

# Finally we fit the model for the chosen # of leaves
model3.fit = prune.tree(tree.mod, best=optSize)
plot(model3.fit)
text(model3.fit, pretty=0)




#e######### - LDA


#install.packages('MASS')
library(MASS)
model4.fit <- lda(Purchase~., data=Orange_Data[train_rows,])

#Evaluate this model on the training set
pred_train4 <- predict(model4.fit, data=Orange_Data[train_rows,])$class
table(pred_train4,Orange_Data$Purchase[train_rows])
#Error Rate on training
mean(pred_train4!=Orange_Data$Purchase[train_rows])





#f######### - KNN


#We first scale the data because the week covariate has huge numbers compared to the rest
Scaled_Orange_Data = scale(Orange_Data[,-1], center = TRUE, scale = TRUE)
Scaled_Orange_Data = cbind(Orange_Data[,1],Scaled_Orange_Data)

train_data = Scaled_Orange_Data[train_rows,]
#We start finding the best k from the vector 1:100. We set the seed to be able to replicate in the future
set.seed(1337)
library(class)
k.vec = 1:100
folds = 5
err.mat.knn = matrix(0, ncol = length(k.vec), nrow = folds)
for (i in 1:folds) {
  cv.ind = sample(1:nrow(train_data), .8*nrow(train_data))
  train.cv = train_data[cv.ind,]
  test.cv = train_data[-cv.ind,]
  for (j in 1:length(k.vec)) {
    knn.pred = knn(train.cv[,-1], test.cv[,-1], train.cv[,1], prob = TRUE, k=k.vec[j])
    err.mat.knn[i,j] = sum(knn.pred != test.cv[,1])/length(knn.pred) 
  }
}
k.opt = k.vec[which.min(apply(err.mat.knn, 2, mean))]

#We compute the classification success on the training data (1-error)
#Of course we need to divide it again in train and test, but we use the right k = k.opt
ind = sample(1:nrow(train_data), .8*nrow(train_data))
train.knn = train_data[ind,]
test.knn = train_data[-ind,]
knn.pred = knn(train.knn[,-1], test.knn[,-1], train.knn[,1], prob = TRUE, k=k.opt)
print(mean(knn.pred != test.knn[,1]))



#g######## - VALIDATION

#LASSO
pred2 = predict(model2.fit, x[validate_rows,],type="class")
table(pred2,y[validate_rows])
lasso_error = mean(pred2 != y[validate_rows])
print(lasso_error)

#Decision Tree
pred3 = predict(model3.fit, newdata=Orange_Data[validate_rows,])
pred3 = round(pred3, digits=0)
table(pred3,y[validate_rows])
tree_error = mean(pred3 != y[validate_rows])
print(tree_error)

#LDA
pred4 <- predict(model4.fit, newdata=Orange_Data[validate_rows,])$class
table(pred4,y[validate_rows])
lda_error = mean(pred4 != y[validate_rows])
print(lda_error)

#KNN
train.knn = Scaled_Orange_Data[train_rows,]
test.knn = Scaled_Orange_Data[validate_rows,]
pred5 = knn(train.knn[,-1], test.knn[,-1], train.knn[,1], prob = TRUE, k=k.opt)
knn_error = mean(pred5 != test.knn[,1])
print(knn_error)


# The minimum error in the validaiton set corresponds to LDA (15.5%)
# Lasso and Decision tree yield very similar results (16.7% - 17.5%)
# KNN is clearly the least accurate (19.8%): a lot of unuseful covariates




# h ######### - FIT FINAL MODEL
total_train_rows = ifelse(RowAssignment == "train" || RowAssignment == "validate",TRUE,FALSE)
finalmodel.fit <- lda(Purchase~., data=Orange_Data[total_train_rows,])

#Evaluate this model on the test set
predfinal <- predict(finalmodel.fit, newdata=Orange_Data[test_rows,])$class
table(predfinal,Orange_Data$Purchase[test_rows])
mean(predfinal != Orange_Data$Purchase[test_rows])

#The error was reduced from 15.5% to 15%



# i #########
#We will use our best model: LDA

predfinal <- predict(finalmodel.fit, newdata=Orange_Data[test_rows,])$class
table_results = table(predfinal,Orange_Data$Purchase[test_rows])

cost = 1*table_results[1,2] #Expected 0, real 1: promotion wasted
revenue = 3*table_results[1,1] #Expected 0, real 0: positive effect
payoff = revenue - cost
print(payoff)
