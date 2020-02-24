# Predicting the GDP of Countries
library(keras)
library(ggplot2)
library(scales)

#link to download data set as a zip: https://www.kaggle.com/fernandol/countries-of-the-world/download
#link to download the data as csv: https://www.kaggle.com/fernandol/countries-of-the-world/download/CqS6Seu7dVMoA1loYfvL%2Fversions%2FBD2NiRkiiKuFttSeyTqj%2Ffiles%2Fcountries%20of%20the%20world.csv?datasetVersionNumber=1
#link to read about the data: https://www.kaggle.com/fernandol/countries-of-the-world




RawData<- read.csv("countries of the world.csv")
data<- RawData
head(data)
names(data)
new_names<-c("Country","Region","Population","Area","PopDensity","Coastline","Netmigration","Infantmortality","GDP","Literacy","Phones","Arable","Crops","Other","Climate","Birthrate","Deathrate","Agriculture","Industry","Service")
names(data)<- new_names
names(data) #much nicer
rm(new_names)

############### Exploring the data


which(is.na(data) , T)
# The GDP is missing for Western Sahara so lets delete this country
data<- data[-c(224),]
# More values are ,missing but they do not appear as "NA"
# example: the country "Andora" has a no value under "agriculture"
#but it is not an NA
data[4,18]

data[,1]<- 1:226 #replace the names of countries by number
rownames(data) <- NULL

test<- data$Region
head(test)
# The variable Region contains 11 levels, each level is a represented as 
#a character. For ease and better performance, Region has to be turned into
#a matrix:
data$Region<- to_categorical(as.numeric(data$Region)-1)
data$Region
rm(test)

# To find missing data stored as "" instead of "NA"
for(i in 1:20){
 cat( i,"\t",which(data[,i] == ""),"\n")
}

data[which(data[,20] == ""), 20] <- NA


# To deal with all the NA, it will be easier to turn the data into numeric
str(data) #most of the observations are stored as Factors...

#all "integer" to "numeric"
for(i in c(1, 3, 4, 9)){
  b<- lapply(data[,i], as.numeric)
  b<- as.data.frame(b)
  b<-t(b)
  b<- as.numeric(b)
  data[,i]<-b
}
str(data)


#all "factor" to "numeric"

b<- c(5:8,10:20)

for (i in b){
  data[,i]<- as.numeric(data[,i])
}
str(data)
rm(b, i)
# Now all the missing values are called NA and the data is stored as numeric
# Let's deal with the missing values. Should we replace them
#with the mean or the median?

summary(data) #spots the column with "NA"

# Replace every NA with the mean of its column EXECPT for climate
# For climate, replace with the median beacause climate has
#qualitative values and not a quantitative so replacing it with the mean will 
#most likely introduce a new variable which does not make any sense.

data$Climate[c(which(is.na(data$Climate == T)))]<- median(data$Climate, na.rm = T)
summary(data)

# The only NA left are in services.

data$Service[which(is.na(data$Service == TRUE))]<- mean(data$Service, na.rm = T)

summary(data)
which(is.na(data) == TRUE)# No NA left !!! ^^
str(data) # Everything is a numeric which is great!

# Place the responce variable "GDP" as the last variable
head(data,2)
names(data)
data[,21]<- data[,9]
head(data,2)
data[,9]<- data[,20]
data[,20]<- data[,21]
data<- data[,1:20]
names(data)[20]<- "GDP"
names(data)[9]<- "Services"
head(data,2)

# Let's look for outliers using boxplot

par(mfrow = c(4,5))
for (i in 1:20){
  boxplot(data[,i], main = names(data)[i] , boxwex=0.1)
}
par(mfrow = c(1,1))
# If you get "Error in plot.new() : figure margins too large"
#make your "Plots" window larger

for(i in 1:20){
  cat( i, "\t", names(data)[i],"\t",  boxplot.stats(data[,i])$out, "\n") 
}

b <- boxplot.stats(data$Region)$out 
length(b)

# Each column of Region contains so many 0's that the few 1's are considred 
#outliers. As a prouf, the number of outliers in region equals the number
#of obvervation in data.
# Climate has outliers but it does not matter because
#they are qualitative values.
# No disturbing outliers were found.

#Just by curriosity, how is the GDP distributed, is it a normal distribution ?
ggplot(data = data, aes(data$GDP))+
  geom_histogram()
#nothing close to a normal distribution...

# Let's get rid of countries:
data<- data[,-c(1)]

str(data)

#for scaling purposes, lat's divide the GDP by a 100
range(data$GDP)
data$GDP<- data$GDP/100
range(data$GDP)
# Should I take the normalize the GDP?


##############################################
# The data can now be splits into training and test set

n<- nrow(data)
p<- ncol(data)-1
set.seed(1)
b<- sample(n, .8*n)

trainSet<- data[b,1:p]
trainLabels<- data[b, p+1]

testSet<- data[-b, 1:p]
testLables<- data[-b, p+1]

# Normalise the trainSet and use those values to normalize the test set.


trainSet<- scale(trainSet)
col_mean_train<- attr(trainSet, "scaled:center")
col_std_train<- attr(trainSet, "scaled:scale")

testSet<- scale(testSet,
                center = col_mean_train,
                scale = col_std_train)

# A little more cleaning and we are DONE !!
rm(i, n, p, b, col_mean_train, col_std_train, col_mean_train)


##########################################################################
#-------------------------------------------------------------------------
##########################################################################


# ANN with Keras: build the model - compile - train/fit - Predict

#build the model:

tensorflow::tf$random$set_seed(2)
  
early_stop <- callback_early_stopping(monitor = "val_loss",
                                      patience = 5)

model<- keras_model_sequential(layers = list(
  layer_dense(units = 15, activation = "relu", input_shape = dim(trainSet)[2]),
  layer_dense(units = 10, activation = "relu", input_shape = 15),
  layer_dense(units = 10, activation = "relu", input_shape = 10),
  layer_dense(units = 10, activation = "relu", input_shape = 10),
  layer_dense(units = 10, activation = "relu", input_shape = 10),
  layer_dense(units = 5, activation = "relu", input_shape = 10),
  layer_dense(units = 1, activation = "linear", input_shape = 5)
))

compile(model,
        loss = "MSE",         #"MSE" Because we are doing regeression
        optimizer = "Adam")   # I read that "Adam" works well with regression

history_with_E_stop <- fit(model,
               trainSet, trainLabels,
               epochs = 55,
               batch_size = 8,
               validation_split = 0.2,
               callbacks = early_stop)


plot(history_with_E_stop, smooth = F)
history_with_E_stop 

test_predictions <- predict(model, testSet)
test_predictions[ , 1]
head(data.frame(true=testLables,
                  predicted=test_predictions))

TrueVSPredicted<- data.frame(true=testLables,
                             predicted=test_predictions)
#let's get the mean absolute error:
MAE<- (1/ncol(TrueVSPredicted))*sum(abs(TrueVSPredicted[,1] - TrueVSPredicted [,2]))
MAE # It is really bad...

ggplot(TrueVSPredicted, aes(true, predicted))+
  geom_point()+
  geom_abline(intercept = 0, slope = 1)
  #Our model mostly outputs a lower GDP

evaluation<- evaluate(model, testSet, testLables)
evaluation
# I tried taking the z-score of the GDP but it did not help..
# I tried many different models, with different number of hidden neuron, different activation
#function and batch size... 
# I found that having more layers makes converging faster 
# Usualy less than 50 epoch compare to a few hundreds with one or two layer.
# But the final result is still crap no matter what ...



######
###### My model is crap !!!
######

# what to do ?
#let's have a look at the data
# Rescale the data to the bound of the activation funciton

# Let's have a look at the ranges in our data

testSetInfo<- data.frame(Range = numeric(0), Min = numeric(0), Max = numeric(0))


for(i in 1:29){
  testSetInfo[i,2]<-range(testSet[,i])[1]
  testSetInfo[i,3]<-range(testSet[,i])[2]
  testSetInfo[i,1]<- abs(testSetInfo[i,2]-testSetInfo[i,3])
}
row.names(testSetInfo)<- c( "Region.1",   "Region.2",   "Region.3",   "Region.4",   "Region.5",   "Region.6","Region.7",   "Region.8",   "Region.9",  "Region.10", "Region.11", "Population","Area", "PopDensity",  "Coastline", "Netmigration", "Infantmortality",  "Services",   "Literacy",     "Phones",     "Arable",      "Crops",      "Other",    "Climate",    "Birthrate",   "Deathrate", "Agriculture",   "Indusry")

head(testSetInfo)

mean(testSetInfo$Range)
mean(testSetInfo$Min)
mean(testSetInfo$Max)

# If we are using relu as an activation function, our data must range
#from -1 to 1
testSet2<- testSet
testSet2<- rescale(testSet2, to = c(0, 1))
trainSet2<- rescale(trainSet, to = c(0, 1))
#
#
# Run the model one more time with the rescaled data
#
#
#


tensorflow::tf$random$set_seed(2)

early_stop <- callback_early_stopping(monitor = "val_loss",
                                      patience = 20)

model<- keras_model_sequential(layers = list(
  layer_dense(units = 3, activation = "relu", input_shape = dim(trainSet)[2]),
#  layer_dense(units = 3, activation = "relu", input_shape = 3),
  layer_dense(units = 1, activation = "linear", input_shape = 3)
))

compile(model,
        loss = "MSE",
        optimizer = "RMSprop") #Let's try something different than "Adam" in case it could help.

history_with_E_stop <- fit(model,
                           trainSet2, trainLabels,
                           epochs = 4000,
                           batch_size = 8 ,
                           validation_split = 0.2,
                           callbacks = early_stop)

plot(history_with_E_stop, smooth = F)
history_with_E_stop 

eval.results <- evaluate(model,
                         testSet2,
                         testLables,
                         verbose = 0)
eval.results

test_predictions <- predict(model, testSet2)
test_predictions[ , 1]
head(data.frame(true=testLables,
                predicted=test_predictions))

TrueVSPredicted<- data.frame(true=testLables,
                             predicted=test_predictions)
# Let's get the mean absolute error:
MAE<- (1/ncol(TrueVSPredicted))*sum(abs(TrueVSPredicted[,1] - TrueVSPredicted [,2]))
MAE

ggplot(TrueVSPredicted, aes(true, predicted))+
  geom_point()+
  geom_abline(intercept = 0, slope = 1)


#little cleaning
rm(TestSetRange, i)


##
##
## The model is still crap. Let's try a linear model.
##
##

# The dataset for the linear model will just be normalised

lmTestData<- as.data.frame(cbind(testSet, testLables))
names(lmTestData)[29]<- "GDP"
head(lmTestData, 2)

lmTrainData<- as.data.frame(cbind(trainSet, testLables))
names(lmTrainData)[29]<- "GDP"
head(lmTrainData, 2)

lmModel<- lm(data = lmTrainData, GDP~.)

results<- predict.lm(lmModel, lmTestData)
head(results)
lmMAE<- (1/nrow(lmTestData))*sum(abs(results - lmTestData$GDP))
lmMAE

lmTestResults<- data.frame(Actual = results, Predicted = lmTestData$GDP)

ggplot(lmTestResults, aes(results, lmTestData$GDP))+
  geom_point()+
  geom_abline(intercept = 0, slope = 1)

# The model is still pretty bad !!!
# What is wrong? Is the GDP unpredictable ? 




 

