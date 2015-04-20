# PML Project: Human Activity Recognition
Ray Jones  
26th April, 2015  


### Executive Summary
This document has been prepared as a submission in response to the Project requirements for the Practical Machine Learning (PML) module of the Coursera/Johns Hopkins Data Science Specialization. The aim of the project was to train a machine learning (ML) algorithm to analyse a data set consisting of quantified self movement measurements collected from accelerometers worn by a number of people during exercises. After initialization and pre-processing of the data set, a number of different ML algorithms were tried and their performances assessed. On this basis, the best ML algorithm was selected and subjected to cross-validation. The selected ML algorithm was then used to generate a prediction of how well the exercises were performed for $20$ blind test cases. This prediction was submitted for grading and a $100 \%$ accuracy level was achieved using the randomForest ML algorithm.

### Exploratory Analysis & Data Pre-processing
The data sets for this project are available from https://d396qusza40orc.cloudfront.net/predmachlearn/ - they consist of pml-training.csv which includes the training/testing data and pml-testing.csv which includes the 20 blind test cases for the project submission. Full instructions on this project are available at https://class.coursera.org/predmachlearn-013/human_grading/view/courses/973548/assessments/4/submissions - the collection and original analysis of this data set are described in a paper[^1] available online from http://groupware.les.inf.puc-rio.br/har .

[^1]: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

The code chunk below loads the various required libraries and then sets the random number seed in order to make the processing reproducible. Following this, it then sets the working directory and loads the two data files required.

```r
## Install the required libraries
library(lattice, quietly=T); library(ggplot2, quietly=T); library(caret, quietly=T)
library(parallel, quietly=T); library(foreach, quietly=T); library(iterators, quietly=T)
library(doParallel, quietly=T); library(randomForest, quietly=T); library(ipred, quietly=T)
library(plyr, quietly=T); library(survival, quietly=T); library(splines, quietly=T)
library(gbm, quietly=T); library(MASS, quietly=T); library(klaR, quietly=T)
library(kernlab, quietly=T); library(caTools, quietly=T); library(rpart, quietly=T)

## Set the random number seed to make this all reproducible
set.seed(1221)

## Set the working directory and then load the required data files
dirPath             <- "~/Documents/Data Science Specialisation/8 - Practical Machine Learning/Project"
setwd(dirPath)
dataSet 			<- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
submissionCases 	<- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
```
The next code chunk prints out the dimensions of the two data sets - (i) dataSet, which contains the data for the training and cross-validation of the ML algorithms and (ii) submissionCases, which contains the $20$ individual observations used for the blind prediction performance testing submitted for grading. This chunk also examined the structures of the two data sets but these lines were subsequently commented out as the outputs were very verbose. In the original analysis the data were treated as time series. However, this was not an option for this exercise as the $20$ blind test cases were supplied out of context (ie without histories or future samples) so the analysis reported here was done on the basis of the instantaneous measurement values and not via time-slicing. 

The next step was to remove the first seven variables from each data set, as they contain values added to the raw measurement data during the original analysis which were not relevant to the analysis here. Following this, variables with incomplete (missing) measurements were identified and removed. In order to reduce the computational load involved in ML algorithm training, pairs of strongly correlated variables were next identified using the findCorrelations() function and one of each pair was removed from the data set (since one of each pair was redundant as a predictor due to the correlation). The dimensions of the resulting data sets were then printed. As a final check before data partitioning, an assessment was carried out to establish that no zero or near zero variance variables were present in the final filtered data sets - this was confirmed. Finally, the code chunk below was used to randomly partition the data set in to training ($60 \%$) and testing ($40 \%$) sub-sets (to allow training, in-sample and out-of-sample (cross-validation) testing) and then print the dimensions of each. 

```r
## Exploratory data analysis
cbind(c("","submissionCases", "dataSet"), rbind(c("Obs","Vars"), dim(submissionCases), dim(dataSet)))
```

```
##      [,1]              [,2]    [,3]  
## [1,] ""                "Obs"   "Vars"
## [2,] "submissionCases" "20"    "160" 
## [3,] "dataSet"         "19622" "160"
```

```r
## str(submissionCases); str(dataSet)

## Remove irrelevant and empty variables
dataSet 			<- dataSet[,-c(1:7)]
submissionCases 	<- submissionCases[,-c(1:7)]
dataSet 			<- dataSet[,colSums(is.na(dataSet)) == 0]
submissionCases 	<- submissionCases[,colSums(is.na(submissionCases)) == 0]

## Find strongly correlated variables & remove from data sets
correlations 		<- findCorrelation(cor(dataSet[,1:52]), cutoff=0.8)
dataSet 			<- dataSet[,-c(correlations)]
submissionCases 	<- submissionCases[,-c(correlations)]
cbind(c("","submissionCases", "dataSet"), rbind(c("Obs","Vars"), dim(submissionCases), dim(dataSet)))
```

```
##      [,1]              [,2]    [,3]  
## [1,] ""                "Obs"   "Vars"
## [2,] "submissionCases" "20"    "41"  
## [3,] "dataSet"         "19622" "41"
```

```r
## str(submissionCases); str(dataSet)

## Confirm absence of zero & near zero variance terms
dataSet_nzv 		<- nearZeroVar(dataSet, saveMetrics = TRUE)
submissionCases_nzv <- nearZeroVar(submissionCases, saveMetrics = TRUE)
cbind(sum(as.numeric(dataSet_nzv$zeroVar)), sum(as.numeric(dataSet_nzv$nzv)),
	  				sum(as.numeric(submissionCases_nzv$zeroVar)),sum(as.numeric(submissionCases_nzv$nzv)))
```

```
##      [,1] [,2] [,3] [,4]
## [1,]    0    0    0    0
```

```r
## Partition the dataSet into 60% training and 40% test data sets
intrain 			<- createDataPartition(dataSet$classe, p = 0.60, list = FALSE)
training 			<- dataSet[intrain,]
testing 			<- dataSet[-intrain,]
cbind(c("","testing", "training"), rbind(c("Obs","Vars"), dim(testing), dim(training)))
```

```
##      [,1]       [,2]    [,3]  
## [1,] ""         "Obs"   "Vars"
## [2,] "testing"  "7846"  "41"  
## [3,] "training" "11776" "41"
```
As can be seen above, after pre-processing the data sets consist of 41 variables. In the training & testing sets this includes the $classe variable which contains the truth for the quality of exercise performance. In submissionCase the $classe variable is absent, replaced by a test id no. (ie for the submissionCases the actual truth data is unknown a-priori and hence these test cases are blind).  

### Model Training
As the next step, the code chunk below was used to train a total of seven different methods implemented within the R caret package using the testing sub-set of the partitioned test data set. These methods were selected to cover a range of different approaches and so maximise the probability of discovering an optimal approach. Since this was computationally demanding, the code first turns on parallel processing and then trains each of the different ML algorithms in turn before finally turning off parallel processing again. This code chunk also caches the results since the total execution elapsed time exceeds $2$ hours (and therefore it was not practical to re-run this chunk many times during the development of the later stages of this project). The seven methods selected for training and later cross comparison of results were:  

* Random Forest (rf)  
* Recursive Partioning & Regression Trees (rpart)  
* Bagged Trees (treebag)  
* Boosted Trees (gbm)  
* Linear Discriminant Analysis (lda)  
* Support Vector Machines - Linear (svmLinear)  
* Support Vector Machines - Radial (svmRadial)  
* Logit Boost Classification (LogitBoost)  

For further discussion of the details of any of these ML algorithms the reader is referred to the relevant R documentation (typing ?method_name at the R command prompt - after the relevant libraries have been loaded).

```r
## Set up parallel processing
cluster 			<- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

## Train candidate models
model_rf 			<- train(classe ~ ., data = training, method = "rf")
model_rpart			<- train(classe ~ ., data = training, method = "rpart")
model_treebag		<- train(classe ~ ., data = training, method = "treebag")
model_gbm 			<- train(classe ~ ., data = training, method = "gbm")
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2104
##      2        1.4788             nan     0.1000    0.1496
##      3        1.3870             nan     0.1000    0.1114
##      4        1.3179             nan     0.1000    0.1046
##      5        1.2534             nan     0.1000    0.0780
##      6        1.2044             nan     0.1000    0.0643
##      7        1.1629             nan     0.1000    0.0714
##      8        1.1187             nan     0.1000    0.0552
##      9        1.0842             nan     0.1000    0.0473
##     10        1.0552             nan     0.1000    0.0554
##     20        0.8255             nan     0.1000    0.0349
##     40        0.5754             nan     0.1000    0.0121
##     60        0.4520             nan     0.1000    0.0059
##     80        0.3703             nan     0.1000    0.0069
##    100        0.3086             nan     0.1000    0.0047
##    120        0.2614             nan     0.1000    0.0026
##    140        0.2234             nan     0.1000    0.0017
##    150        0.2089             nan     0.1000    0.0015
```

```r
model_lda			<- train(classe ~ ., data = training, method = "lda")
model_svml			<- train(classe ~ ., data = training, method = "svmLinear")
model_svmr 			<- train(classe ~ ., data = training, method = "svmRadial")
model_LogitBoost 	<- train(classe ~ ., data = training, method = "LogitBoost")

## turn off parallel processing
stopCluster(cluster)
```

### Cross Validation & Expected Out Of Sample Error
The next code chunk compared the performances of the various methods used by assembling and printing a table of the in-sample accuracies achieved by each of the seven after using the trained models to predict the $classe results for the testing sub-set and comparing them to the actual $classe variables using the caret confusionMatrix routine. 

```r
## Compare & print candidate model training accuracy results
modelFit 			<- c("rf", "rpart","treebag","gbm", "lda", "svmLinear", "svmRadial", "LogitBoost")

modelRes 			<- cbind(modelFit, round(rbind(confusionMatrix(predict(model_rf, training), 
														   						training$classe)$overall[1:4],
	  					confusionMatrix(predict(model_rpart, training), training$classe)$overall[1:4],
	  					confusionMatrix(predict(model_treebag, training), training$classe)$overall[1:4],
	  					confusionMatrix(predict(model_gbm, training), training$classe)$overall[1:4],
	  					confusionMatrix(predict(model_lda, training), training$classe)$overall[1:4],
	  					confusionMatrix(predict(model_svml, training), training$classe)$overall[1:4],
	  					confusionMatrix(predict(model_svmr, training), training$classe)$overall[1:4],
	  					confusionMatrix(predict(model_LogitBoost, training), training$classe)$overall[1:4]),5))
modelRes
```

```
##      modelFit     Accuracy  Kappa     AccuracyLower AccuracyUpper
## [1,] "rf"         "1"       "1"       "0.99969"     "1"          
## [2,] "rpart"      "0.49924" "0.34641" "0.49016"     "0.50831"    
## [3,] "treebag"    "0.99966" "0.99957" "0.99913"     "0.99991"    
## [4,] "gbm"        "0.96527" "0.95606" "0.9618"      "0.9685"     
## [5,] "lda"        "0.64232" "0.54782" "0.63359"     "0.65099"    
## [6,] "svmLinear"  "0.70958" "0.63048" "0.70129"     "0.71777"    
## [7,] "svmRadial"  "0.92425" "0.90398" "0.91933"     "0.92897"    
## [8,] "LogitBoost" "0.87421" "0.84003" "0.86754"     "0.88066"
```
From the results presented in the table above, it is clear that the random forest (rf) and bagged trees (treebag) algorithms gave the best results with in-sample accuracies at or near $100 \%$. The boosted trees (gbm) and support vector machones - radial (svmr) algorithms both gave in-sample accuracies in excess of $90 \%$ while all others fell in the range $49 - 88 \%$. Since the highest accuracy was achieved by the random forest algorithm (the same algorithm used in the original research paper), theis method was selected for further testing at this point.

Initially the detailed in-sample test results for the random forest algorithm were printed out by the following code chunk using the confusionMatrix routine. Then the trained random forest model was used to predict the $classe results for the testing sub-set of the data and details of the out of sample performance (cross-validation) were printed.  Finally a bar chart graphic comparing the in sample and out of sample performances was prepared to allow easy comparison of the results.

```r
## Print out details of selected model training performance
prediction 			<- predict(model_rf, training)
cm_training 		<- confusionMatrix(prediction, training$classe)
cm_training
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
## Use selected model to predict testing results and assess performance
prediction 			<- predict(model_rf, testing)
cm_testing 			<- confusionMatrix(prediction, testing$classe)
cm_testing
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2222   10    0    1    0
##          B    9 1500   15    0    0
##          C    1    8 1345   13    2
##          D    0    0    8 1271    1
##          E    0    0    0    1 1439
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9912          
##                  95% CI : (0.9889, 0.9932)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9889          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9955   0.9881   0.9832   0.9883   0.9979
## Specificity            0.9980   0.9962   0.9963   0.9986   0.9998
## Pos Pred Value         0.9951   0.9843   0.9825   0.9930   0.9993
## Neg Pred Value         0.9982   0.9972   0.9964   0.9977   0.9995
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2832   0.1912   0.1714   0.1620   0.1834
## Detection Prevalence   0.2846   0.1942   0.1745   0.1631   0.1835
## Balanced Accuracy      0.9968   0.9922   0.9897   0.9935   0.9989
```

```r
## Plot comparison charts for training and testing performance of selected model
par(mfrow=c(1,2))
plot(cm_training$table, col = cm_training$byClass, main = paste("RF Model In Sample Accuracy =", 
																round(cm_training$overall['Accuracy'], 5)))
plot(cm_testing$table, col = cm_testing$byClass, main = paste("RF Model Out Of Sample Accuracy =", 
																round(cm_testing$overall['Accuracy'], 5)))
```

<img src="Project_files/figure-html/validation-1.png" title="" alt="" style="display: block; margin: auto;" /><p class="caption">*Figure 1: Selected RF Model Performance*</p>
The detailed in sample results for the random forest method show the very high accuracy performance achieved - $100 \%$. As expected, this performance was lower for the out of sample (cross-validation) testing but nevertheless exceeded $99.1 \%$ accuracy. At this accuracy level, the probability of achieving a perfect score for the $20$ blind test cases was computed as $0.9912^{20} \approx 0.8380$. But given that $3$ attempts were allowed, the probability that at least one of those attempts would achieve a perfect score was computed as $1-(1-0.8380)^{3} \approx 0.9957$, ie a $99.6 \%$ probability of achieving a perfect score. On this basis, the decision was taken to proceed with the blind test case submission. Examining the figure above shows these very high levels of prediction accuracy graphically. On the the LHS the bar chart shows the perfect results acheived in sample (hence the bars are shown filled in) compared to the RHS where the out of sample results do show a small (but acceptable) number of errors.

### Rationale For Design Choices & Submission
The text above describes the rationales for various design choices made, including:  
* Avoiding a time slice based approach  
* Filtering of the data sets  
* Partitioning of the data sets  
* Training of multiple ML algorithms  
* Caching of the computations   
* Selecting the ML algorithm for use with the blind test cases  
* Computing the likely performance against the blind test cases and justifying the decision to proceed.

But the ultimate rationale for these choices is the actual performance achieved in the grading of the $20$ blind test cases. The following code chunk was used to predict the (unknown) $classe results for each of the $20$ test cases, print them out and finally prepare the correctly formatted files for submission.

```r
## Use selected model to predict submission results
submission 			<- predict(model_rf, submissionCases)
submission
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
## Supplied function for creating results files
pml_write_files 	<- function(x){
  n 				<- length(x)
  for(i in 1:n){
    filename 		<- paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

## Create results files
pml_write_files(submission)
```
The files prepared above were then submitted to the grading process. A perfect score (20 out of 20 correct) was achieved on the first submission.

### Conclusions
The data sets required were loaded and pre-processed. A number of candidate ML algorithms were trained and assessed. The random forest method was selected and out of sample results assessed. Finally the blind test cases were computed and submitted, achieving a perfect score. These activities were described in the document above. Therefore, all the requirements for this project have been met.
