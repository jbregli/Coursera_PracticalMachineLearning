library(caret);library(kernlab)
set.seed(32323)

# Load the test file:
dataset  <- read.csv("pml-training.csv", na.strings=c("NA",""), strip.white=TRUE)
dim(dataset)

# Cleaning the data:
isNA <- apply(dataset, 2, function(x) { sum(is.na(x)) })

dataset <- subset(dataset[, which(isNA == 0)], 
                    select=-c(X, user_name, new_window, num_window, 
                              raw_timestamp_part_1, raw_timestamp_part_2, 
                              cvtd_timestamp))
dim(dataset)

# Spliting the dataset:
inTrain <- createDataPartition(dataset$classe, p=0.75, list=FALSE)
train_set <- dataset[inTrain,]
valid_set <- dataset[-inTrain,]

# Test 1: Random forest classifier:
ctrl <- trainControl(allowParallel=TRUE, method="cv", number=4)
model <- train(classe ~ ., data=train_set, model="rf", trControl=ctrl)
predictor <- predict(model, newdata=valid_set)

# Error on valid_set:
sum(predictor == valid_set$classe) / length(predictor)
confusionMatrix(valid_set$classe, predictor)$table

# Classification for test_set:
dataset_test <- read.csv("pml-testing.csv", na.strings=c("NA",""), strip.white=T)
dataset_test <- subset(dataset_test[, which(isNA == 0)], 
                        select=-c(X, user_name, new_window, num_window,
                                  raw_timestamp_part_1, raw_timestamp_part_2,
                                  cvtd_timestamp))

predict(model, newdata=dataset_test)

# Test 2: SVM:
# Most important variable for the ramdom forest predictor:
varImp(model)

# Let us train the SVM on the dataset reduce to those ten variables
dataset_small <- subset(dataset, 
                         select=c(roll_belt, pitch_forearm, yaw_belt,
                                  magnet_dumbbell_y, pitch_belt, 
                                  magnet_dumbbell_z, roll_forearm,
                                  accel_dumbbell_y, roll_dumbbell,
                                  magnet_dumbbell_x,classe))

svm <- train(classe ~ ., data=dataset_small[inTrain,], model="svm", trControl=ctrl)

predictor_svm <- predict(svm, newdata=valid_set)

# Error on valid_set:
sum(predictor_svm == valid_set$classe) / length(predictor_svm)
confusionMatrix(valid_set$classe, predictor_svm)$table

# Classification for test_set:
predict(model, newdata=dataset_test)
