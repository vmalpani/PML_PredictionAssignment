library(caret)
library(randomForest)
library(knitr)
opts_chunk$set(cache=TRUE)

training_data = read.csv('./data/pml-training.csv')
testing_data = read.csv('./data/pml-testing.csv')

set.seed(1234)
train_idx <- createDataPartition(training_data$classe,p=0.75,list=FALSE)
training_set <- training_data[train_idx,]
validation_set <- training_data[-train_idx,]

nzv_cols <- nearZeroVar(training_set)
training_set <- training_set[-nzv_cols]
validation_set <- validation_set[-nzv_cols]
testing_set <- testing_data[-nzv_cols]

numeric_cols <- which(lapply(training_set,class) %in% c('numeric'))

preProcessingModel <- preProcess(training_set[,numeric_cols], method=c('knnImpute'))

preProcessedTraining <- cbind(training_set$classe, predict(preProcessingModel,training_set[,numeric_cols]))
names(preProcessedTraining)[1] <- 'classe'
preProcessedTraining[is.na(preProcessedTraining)] <- 0

preProcessedValidation <- cbind(validation_set$classe, predict(preProcessingModel,validation_set[,numeric_cols]))
names(preProcessedValidation)[1] <- 'classe'
preProcessedValidation[is.na(preProcessedValidation)] <- 0

preProcessedTesting <- cbind(testing_set$problem_id, predict(preProcessingModel,testing_set[,numeric_cols]))
preProcessedTesting[is.na(preProcessedTesting)] <- 0

Different algorithms were tried but finally we fixed upon training a random forest model.
randomForestModel <- randomForest(classe~., preProcessedTraining)

trainingPredictions <- predict(randomForestModel, preProcessedTraining)
print(mean(trainingPredictions == preProcessedTraining$classe))

validationPredictions <- predict(randomForestModel, preProcessedValidation)
print(mean(validationPredictions == preProcessedValidation$classe))

testingPredictions <- predict(randomForestModel, preProcessedTesting)
print(testingPredictions)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(as.character(testingPredictions))
