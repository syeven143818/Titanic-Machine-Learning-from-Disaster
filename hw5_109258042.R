# use machine learning to create a model that predicts 
# which passengers survived the Titanic shipwreck

argv <- commandArgs(TRUE)


if (length(argv)==0) {
  stop("You need to add some arguments.")
}


# detect missing flag
if (!"--fold" %in% argv) {
  stop("Miss the flag '--fold'.")
} else if (!"--train" %in% argv) {
  stop("Miss the flag '--train'.")
} else if (!"--test" %in% argv) {
  stop("Miss the flag '--test'.")
} else if (!"--report" %in% argv) {
  stop("Miss the flag '--report'.")
} else if (!"--predict" %in% argv) {
  stop("Miss the flag '--predict'.")
}


i <- 1
while (i < length(argv)) {
  if (argv[i] == "--fold") {
    fold <- as.numeric(argv[i+1])   # convert string to numeric
    i <- i + 1
  } else if (argv[i] == "--train") {
    training_data <- argv[i+1]
    i <- i + 1
  } else if (argv[i] == "--test") {
    testing_data <- argv[i+1]
    i <- i + 1
  } else if (argv[i] == "--report") {
    performance <- argv[i+1]
    i <- i + 1
  } else if (argv[i] == "--predict") {
    my_predict <- argv[i+1]
    i <- i + 1
  } else {
    stop(paste("Unknown flag.", args[i]), call.=FALSE)
  }
  
  i <- i + 1
}


if(!require('rpart')){
  install.packages('rpart', repos = "http://cran.us.r-project.org")
}
library('rpart')


df <- read.csv(training_data, header = T)


# remove some columns
df$PassengerId <- NULL
df$Name <- NULL
df$Ticket <- NULL
df$Cabin <- NULL


# add a column of random variable from uniform distribution
df$gp <- runif(dim(df)[1])


train_accuracy <- c()
vali_accuracy <- c()
test_accuracy <- c()
set_vector <- c()


for (k in 1:fold) {
  
  set_vector <- c(set_vector, paste("fold", k, sep = ""))
  
  # split three sets
  test_set <- subset(df, subset = (df$gp <= k / fold & df$gp > (k - 1) / fold))
  
  if (k == fold) {
    vali_set <- subset(df, subset = (df$gp <= 1 / fold))
    training_set <- subset(df, subset = (df$gp > 1 / fold & df$gp <= (k - 1) / fold))
  } else {
    vali_set <- subset(df, subset = (df$gp <= (k + 1) / fold & df$gp > k / fold))
    training_set <- subset(df, subset = (df$gp > (k + 1) / fold | df$gp <= (k - 1) / fold))
  }
  
  
  
  # model using decision tree
  model <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                 data=training_set, control=rpart.control(maxdepth=4),
                 method="class")
  
  
  
  # make confusion matrix table of training set
  resultframe <- data.frame(truth=training_set$Survived,
                            pred=predict(model, newdata = training_set,type="class"))
  rtab <- table(resultframe)
  
  # compute training accuracy
  train_accuracy <- c(train_accuracy, round(sum(diag(rtab)) / sum(rtab), digits = 2))
  
  
  
  # make confusion matrix table of validation set
  resultframe <- data.frame(truth=vali_set$Survived,
                            pred=predict(model, newdata = vali_set,type="class"))
  rtab <- table(resultframe)
  
  # compute validation accuracy
  vali_accuracy <- c(vali_accuracy, round(sum(diag(rtab)) / sum(rtab), digits = 2))
  
  
  
  # make confusion matrix table of testing set
  resultframe <- data.frame(truth=test_set$Survived,
                            pred=predict(model, newdata = test_set,type="class"))
  rtab <- table(resultframe)
  
  # compute training accuracy
  test_accuracy <- c(test_accuracy, round(sum(diag(rtab)) / sum(rtab), digits = 2))
}

ave_vector <- c("ave.", 
                round(mean(train_accuracy), digits = 2),
                round(mean(vali_accuracy), digits = 2), 
                round(mean(test_accuracy), digits = 2))

new_df <- data.frame(set =  set_vector,
                     training = train_accuracy,
                     validation = vali_accuracy,
                     test = test_accuracy,
                     stringsAsFactors = FALSE)

new_df <- rbind(new_df, ave_vector)

write.csv(new_df, performance, row.names = F, quote = F)


# choose the maximum validation accuracy to select the best prediction model
max_validation_accuracy_index <- which.max(new_df$validation)


test_set <- subset(df, subset = (df$gp <= max_validation_accuracy_index / fold & 
                                   df$gp > (max_validation_accuracy_index - 1) / fold))
if (max_validation_accuracy_index == fold) {
  vali_set <- subset(df, subset = (df$gp <= 1 / fold))
  training_set <- subset(df, subset = (df$gp > 1 / fold & 
                                       df$gp <= (max_validation_accuracy_index - 1) / fold))
} else {
  vali_set <- subset(df, subset = (df$gp <= (max_validation_accuracy_index + 1) / fold &
                                     df$gp > max_validation_accuracy_index / fold))
  training_set <- subset(df, subset = (df$gp > (max_validation_accuracy_index + 1) / fold | 
                                       df$gp <= (max_validation_accuracy_index - 1) / fold))
}


test_set$gp <- NULL
vali_set$gp <- NULL
training_set$gp <- NULL


# combine validation data set and training data set
merge_set <- rbind(vali_set, training_set)

# model using decision tree, data is merge set
model <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
               data=merge_set, control=rpart.control(maxdepth=4),
               method="class")


test_df <- read.csv(testing_data, header = T)
resultframe <- data.frame(PassengerId = test_df$PassengerId,
                          Survived = predict(model,newdata = test_df, type="class"))

write.csv(resultframe, my_predict, row.names = F, quote = F)

