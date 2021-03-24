# SYS 6018 class
# Kaggle Competition 3 - Twitter Self-driving Car Sentiment
# September 26, 2017

# Competition Group 3-7:
# Myron Chang - mc7bk
# Yi Hao - yh8a
# Adrian Mead - atm4rf


# Load in the libraries
library(XML)
library(tm)
library(tidyverse) # For read_csv
library(boot) # For LOOCV
library(MASS) # For ordered logistic regression via polr
library(nnet) # For multinomial model via multinom



################### Initialize the Training Data ###################

# Read in the training data
train_data <- read_csv(file = 'train.csv')

# Initial analysis of training data
summary(train_data)
nrow(train_data)
prop.table(table(train_data$sentiment))
#         1          2          3          4          5 
#   0.02344546 0.11926606 0.61467890 0.18246687 0.06014271 
hist(train_data$sentiment)

# We see from the above code that the distribution of sentiments in the training
# data is uneven. Most of the observations have a sentiment value of 3, some have
# sentiment values of 2 or 4, and even fewer have the extreme sentiments of 1
# or 5.

# Converting the text of the training data to a corpus object
train_tweets <- as.tibble(train_data$text)
twitter_corpus = VCorpus(DataframeSource(train_tweets))

# Compute the initial TF-IDF matrix and inspect sparsity
twitter.tfidf = DocumentTermMatrix(twitter_corpus, control = list(weighting = weightTfIdf))
twitter.tfidf

# We see from the initial TF-IDF matrix that there is incredible sparsity in the
# training data. The document term matrix returned a total of 4,856 distinct
# terms. Also, out of all the entries in the matrix 12,591 are non-sparse,
# compared to 4,751,145 sparse entries. This yields a sparsity percentage of
# roughly 100%.

# Reduce term sparsity by cleaning up the corpus
twitter_corpus.clean = tm_map(twitter_corpus, stripWhitespace)                          # remove extra whitespace
twitter_corpus.clean = tm_map(twitter_corpus.clean, removeNumbers)                      # remove numbers
twitter_corpus.clean = tm_map(twitter_corpus.clean, removePunctuation)                  # remove punctuation
twitter_corpus.clean = tm_map(twitter_corpus.clean, content_transformer(tolower))       # ignore case
twitter_corpus.clean = tm_map(twitter_corpus.clean, removeWords, stopwords("english"))  # remove stop words
twitter_corpus.clean = tm_map(twitter_corpus.clean, stemDocument)

# Recompute TF-IDF matrix
twitter.clean.tfidf = DocumentTermMatrix(twitter_corpus.clean, control = list(weighting = weightTfIdf))
twitter.clean.tfidf

# After cleaning up the corpus through the above code, the cleaned TF-IDF matrix
# shows some improvement. The document term matrix now has 3,292 distinct terms.
# There are 9,967 non-sparse entries in the matrix, compared to 3,219,485 sparse
# entries. This still yields a sparsity percentage of roughly 100%, so we still
# have work to do in cutting down the dimensionality of the training data.

# Remove sparse terms using a 99.5% threshold
twitter.tfidf.sparse = removeSparseTerms(twitter.clean.tfidf, .995)  # remove terms that are absent from at least 99.5% of documents (keep most terms)
twitter.tfidf.sparse

# Using a 99.5% sparsity threshold leaves us with 327 distinct terms, which is
# much more manageable for our analyses. There are 6,001 non-sparse entries and
# 314,786 sparse entries, giving us a sparsity percentage of 98%.

# # Remove sparse terms using a 99% threshold
# twitter.tfidf.99 = removeSparseTerms(twitter.clean.tfidf, .99)  # remove terms that are absent from at least 99% of documents
# twitter.tfidf.99
# 
# # Using a 99% sparsity threshold leaves us with 126 distinct terms. There are 
# # 4,697 non-sparse entries and 118,909 sparse entries, giving us a sparsity 
# # percentage of 96%.

# Remove sparse terms using a 98% threshold
twitter.tfidf.98 = removeSparseTerms(twitter.clean.tfidf, .98)  # remove terms that are absent from at least 98% of documents
twitter.tfidf.98

# Using a 98% sparsity threshold gives 43 distinct terms, with 3,572 non-sparse
# entries and 38,611 sparse entries, giving a sparsity percentage of 92%.

## Remove sparse terms using a 97% threshold
#twitter.tfidf.97 = removeSparseTerms(twitter.clean.tfidf, .97)  # remove terms that are absent from at least 97% of documents
#twitter.tfidf.97

## Using a 97% sparsity threshold gives 26 distinct terms, with 3,161 non-sparse
## entries and 22,345 sparse entries, giving a sparsity percentage of 88%.

## Remove sparse terms using a 96% threshold
#twitter.tfidf.96 = removeSparseTerms(twitter.clean.tfidf, .96)  # remove terms that are absent from at least 96% of documents
#twitter.tfidf.96

## Using a 96% sparsity threshold gives 16 distinct terms, with 2,816 non-sparse
## entries and 12,880 sparse entries, giving a sparsity percentage of 82%.

## Remove sparse terms using a 95% threshold
#twitter.tfidf.95 = removeSparseTerms(twitter.clean.tfidf, .95)  # remove terms that are absent from at least 95% of documents
#twitter.tfidf.95

## Using a 95% sparsity threshold gives 11 distinct terms, with 2,606 non-sparse
## entries and 8,185 sparse entries, giving a sparsity percentage of 76%.

## Remove sparse terms using a 94% threshold
#twitter.tfidf.94 = removeSparseTerms(twitter.clean.tfidf, .94)  # remove terms that are absent from at least 94% of documents
#twitter.tfidf.94

## Using a 94% sparsity threshold gives 6 distinct terms, with 2,339 non-sparse
## entries and 3,546 sparse entries, giving a sparsity percentage of 60%.

## Remove sparse terms using a 70% threshold
#twitter.tfidf.70 = removeSparseTerms(twitter.clean.tfidf, .7)  # remove terms that are absent from at least 70% of documents
#twitter.tfidf.70

## Using a 70% sparsity threshold gives only 2 distinct terms, with 1,834
## non-sparse entries and 128 sparse entries, giving a sparsity percentage of 7%.
## Two terms is way too few to conduct meaningful in-depth analyses with.


# Based on all the above sparsity thresholds we looked at, we decided to use
# the 99.5% and 98% thresholds for our models because they seemed to provide a
# good combination of distinct terms and reduced dimensionality from the raw
# training data.


################### Parametric Linear Models ###################


# Combine training data and 98% TF-IDF matrix into a model-able format
cleaneddata <- as.data.frame(as.matrix(twitter.tfidf.98))   # Uses 98% matrix
newdata <- cbind(train_data,cleaneddata)
newdata2 <- newdata[,-2]  # Drop "text" variable 

# # Linear regression model #1 with corresponding LOOCV
# model.lm <- glm(sentiment~., data=newdata2)
# summary(model.lm)
# anova(model.lm)
# cv.err.lm=cv.glm(newdata2, model.lm)
# cv.err.lm$delta  # 0.6212805 0.6212479
# 
# # Our first parametric model is a multiple linear regression using all 43 terms
# # identified by the 98% TF-IDF matrix as predictor variables. We realize that
# # this model treats the response variable sentiment as an ordinary numeric
# # variable, when it is in fact not, but we wanted to see how this model performs
# # in predicting sentiment. The LOOCV method gives an MSE of 0.6212805. Since
# # the sentiment variable is a classification variable, however, this MSE is
# # difficult to interpret. So, we use an alternate hand-written LOOCV to better
# # compute the accuracy of this model, using the below code.
# 
# # Alternative LOOCV for linear regression model #1
# cvv.lm <- sapply(X = 1:nrow(newdata2), function(X){
#   training_data <- newdata2[-X,]
#   testing_data <- newdata2[X,]
#   cvmodel <- glm(sentiment~., data=training_data)
#   guess <- round(predict(cvmodel, newdata=testing_data))  # 'round' makes the sentiment result an integer, so we can test equality
#   accuracy <- guess == testing_data$sentiment
#   return(accuracy)
# })
# 
# totalaccuracy.lm <- sum(cvv.lm) / length(cvv.lm)
# totalaccuracy.lm    # 0.598369
# 
# # Using the above LOOCV code, we see that the linear regression model predicts
# # the correct sentiment almost 60% of the time.
# 
# # Linear regression model #2 with corresponding LOOCV (only variables with significance markers '.'s or '*'s from first model)
# model.lm2 <- glm(sentiment~cant+come+googl+less+need+think+use+wait+want, data=newdata2)
# summary(model.lm2)
# anova(model.lm2)
# cv.err.lm2 = cv.glm(newdata2, model.lm2)
# cv.err.lm2$delta # 0.5923352 0.5923267
# 
# # Our second linear regression model uses only the variables that were
# # statistically significant at the 90% level from the first model above. This
# # model uses nine predictor variables. The LOOCV from the 'boot' package gives
# # an MSE of 0.5923352 for this model, which is lower than the MSE for the first
# # model above. However, since sentiment is not a numeric variable, the MSE is
# # difficult to interpret, so we again use an alternate LOOCV method.
# 
# # Alternative LOOCV for linear regression model #2
# cvv.lm2 <- sapply(X = 1:nrow(newdata2), function(X){
#   training_data <- newdata2[-X,]
#   testing_data <- newdata2[X,]
#   cvmodel <- glm(sentiment~cant+come+googl+less+need+think+use+wait+want, data=training_data)
#   guess <- round(predict(cvmodel, newdata=testing_data))  # 'round' makes the sentiment result an integer, so we can test equality
#   accuracy <- guess == testing_data$sentiment
#   return(accuracy)
# })
# 
# totalaccuracy.lm2 <- sum(cvv.lm2) / length(cvv.lm2)
# totalaccuracy.lm2    # 0.6146789
# 
# # The LOOCV code shows that the second linear regression model predicts
# # sentiment correctly about 61% of the time. This, along with the lower MSE
# # from above, suggests that the second model with fewer predictors is a better
# # model than the first linear regression model.
# 
# # Linear regression model #3 with corresponding LOOCV (only variables with *s from model 2)
# model.lm3 <- glm(sentiment~cant+googl+need+think+use+wait+want, data=newdata2)
# summary(model.lm3)
# anova(model.lm3)
# cv.err.lm3 = cv.glm(newdata2, model.lm3)
# cv.err.lm3$delta # 0.5925687 0.5925625
# 
# # Our third linear regression model uses only the variables that were
# # statistically significant at the 95% level from the second model. This reduces
# # our model to seven predictor variables. The LOOCV from the 'boot' package gives
# # an MSE of 0.5925687 for this model, which is slightly higher than the MSE for
# # the second linear regression model.
# 
# # Alternative LOOCV for linear regression model #3
# cvv.lm3 <- sapply(X = 1:nrow(newdata2), function(X){
#   training_data <- newdata2[-X,]
#   testing_data <- newdata2[X,]
#   cvmodel <- glm(sentiment~cant+googl+need+think+use+wait+want, data=training_data)
#   guess <- round(predict(cvmodel, newdata=testing_data))  # 'round' makes the sentiment result an integer, so we can test equality
#   accuracy <- guess == testing_data$sentiment
#   return(accuracy)
# })
# 
# totalaccuracy.lm3 <- sum(cvv.lm3) / length(cvv.lm3)
# totalaccuracy.lm3    # 0.617737
# 
# # The third linear regression model predicts sentiment correctly almost 62% of
# # the time. This is better than the second model, even though the 'boot' LOOCV
# # shows a higher MSE. We feel the hand-written LOOCV that predicts actual 
# # accuracy is a better cross-validation method to use for this situation, so in
# # turn we conclude that the third model performs slightly better than the second.
# 
# # Linear regression model #4 with corresponding LOOCV (only variables with ** and *** from model 3)
# model.lm4 <- glm(sentiment~cant+googl+want, data=newdata2)
# summary(model.lm4)
# anova(model.lm4)
# cv.err.lm4 = cv.glm(newdata2, model.lm4)
# cv.err.lm4$delta # 0.5972975 0.5972949
# 
# # Our fourth linear regression model uses only the variables that were
# # statistically significant at the 99% level from the third model. This model
# # has three predictor variables. The LOOCV from the 'boot' package gives an
# # MSE of 0.5972975 for this model, which is higher than those for the second
# # and third models above.
# 
# # Alternative LOOCV for linear regression model #4
# cvv.lm4 <- sapply(X = 1:nrow(newdata2), function(X){
#   training_data <- newdata2[-X,]
#   testing_data <- newdata2[X,]
#   cvmodel <- glm(sentiment~cant+googl+want, data=training_data)
#   guess <- round(predict(cvmodel, newdata=testing_data))  # 'round' makes the sentiment result an integer, so we can test equality
#   accuracy <- guess == testing_data$sentiment
#   return(accuracy)
# })
# 
# totalaccuracy.lm4 <- sum(cvv.lm4) / length(cvv.lm4)
# totalaccuracy.lm4    # 0.6187564
# 
# # The fourth linear regression model predicts sentiment correctly almost 62% of
# # the time. The resulting percentage is slightly higher than that for the third
# # model. Again, even though the 'boot' LOOCV gives a higher MSE, we believe that
# # this fourth model is performing slightly better than the third linear model.
# 
# # Linear regression model #5 with corresponding LOOCV - Model GOOGLE
# model.google <- glm(sentiment~googl, data=newdata2)
# summary(model.google)
# anova(model.google)
# cv.err.google = cv.glm(newdata2, model.google)
# cv.err.google$delta   # 0.6140303 0.6140290
# 
# # For our fifth and final linear regression model, we looked back to the first
# # model containing all 43 terms as predictors and identified the one single term
# # that was the most statistically significant, with the largest absolute
# # t-statistic and smallest p-value. The term was 'googl'. This term is the only
# # predictor variable used in this fifth model. The LOOCV from the 'boot'
# # package gives an MSE of 0.6140303, which is lower than that for the first
# # linear model but higher than all the others.
# 
# # Alternative LOOCV for linear regression model #5
# cvv.google <- sapply(X = 1:nrow(newdata2), function(X){
#   training_data <- newdata2[-X,]
#   testing_data <- newdata2[X,]
#   cvmodel <- glm(sentiment~googl, data=training_data)
#   guess <- round(predict(cvmodel, newdata=testing_data))  # 'round' makes the sentiment result an integer, so we can test equality
#   accuracy <- guess == testing_data$sentiment
#   return(accuracy)
# })
# 
# totalaccuracy.google <- sum(cvv.google) / length(cvv.google)
# totalaccuracy.google    # 0.6116208

# The fifth linear regression model predicts sentiment correctly about 61% of
# the time. This model performs better than the first model, but worse than the
# second, third, and fourth models above. Another interesting note is that the
# fitted model has the equation form "y = 3.0952 + 0.7307x", which in this
# context means this model will only ever predict 3s and 4s for sentiment.



# The above five models are all multiple linear regression models, and assume
# the response variable sentiment is numeric. This is not the case; sentiment
# only has five values: 1, 2, 3, 4, and 5. To account for the fact that
# sentiment is actually a classification variable, we decide to use various
# logistic models to see if they perform better.



# Ordinal logistic regression model #1 (with all 43 terms from 98% TF-IDF matrix)

newdata2$sentiment <- as.factor(newdata2$sentiment)   # Convert sentiment to factor to allow ordinal logistic regression to run

# # Model and summary
# ordinal.model <- polr(sentiment ~ ., data = newdata2, Hess=TRUE)
# summary(ordinal.model)
# 
# # 95% Confidence intervals of variables
# confint(ordinal.model)
# 
# # LOOCV
# cvv.olr <- sapply(X = 1:nrow(newdata2), function(X){
#   training_data <- newdata2[-X,]
#   testing_data <- newdata2[X,]
#   cvmodel <- polr(sentiment ~ ., data=training_data, Hess=TRUE)
#   guess <- predict(cvmodel, newdata=testing_data)
#   accuracy <- guess == testing_data$sentiment
#   return(accuracy)
# })
# 
# totalaccuracy.olr <- sum(cvv.olr) / length(cvv.olr)
# totalaccuracy.olr   # 0.6126402

# This is an ordinal logistic regression model. The response variable sentiment
# is a factor variable with five possible levels. Additionally, these levels are
# ordered, in that 5 is greater than 4, which is greater than 3, which is
# greater than 2, which is greater than 1. These circumstances make ordinal
# logistic regression appropriate to use, because the regression will fit to
# a response that is a factor variable, and it will preserve the ordered
# characteristic of sentiment.
# The model is fitted using the 'polr' command, which stands for proportional
# odds logistic regression. This regression carries an assumption that the log
# odds of each category in the response differ only by a constant, so the log
# odds are proportional.
# We use all of the 43 terms from the TF-IDF matrix created using a 98%
# sparsity threshold as the predictors for this first iteration. The LOOCV
# results show that this ordinal logistic regression model predicts sentiment
# accurately about 61.26% of the time, which is comparable with the performance
# of the multiple linear regression models.

# Ordinal logistic regression model #2 (only include terms where 95% confidence intervals from model #1 did not have zero in them)
ordinal.model2 <- polr(sentiment ~ cant+come+googl+need+think+use+wait+want, data = newdata2, Hess=TRUE)
summary(ordinal.model2)

# 95% Confidence intervals of variables
#confint(ordinal.model2)

# LOOCV
cvv.olr2 <- sapply(X = 1:nrow(newdata2), function(X){
  training_data <- newdata2[-X,]
  testing_data <- newdata2[X,]
  cvmodel <- polr(sentiment ~ cant+come+googl+need+think+use+wait+want, data=training_data, Hess=TRUE)
  guess <- predict(cvmodel, newdata=testing_data)
  accuracy <- guess == testing_data$sentiment
  return(accuracy)
})

totalaccuracy.olr2 <- sum(cvv.olr2) / length(cvv.olr2)
totalaccuracy.olr2    # 0.6146789

# For the second ordinal logistic regression model, we used the 95% confidence
# intervals from the first model above to identify the terms that were
# statistically significant at the 95% level (i.e. the confidence intervals did
# not include zero). This results in eight predictor variables for this second
# logistic model. The LOOCV results show that this second ordinal logistic
# regression model has an accuracy of about 61.47%, which is an improvement over
# the first model. Additionally, the 95% confidence intervals show that all the
# predictors seem to be statistically significant at the 95% level.



# We also decided to look at a multinomial model, which is essentially a
# logistic model that does not assume any ordering. This means that
# theoretically we are losing some information from the ordering of the
# different sentiment values. Nevertheless, we want to see how a multinomial
# model performs in predicting sentiment through LOOCV.

# # Multinomial regression model (using all 43 terms from 98% TF-IDF matrix)
# model.multinom <- multinom(sentiment ~ ., data = newdata2)
# summary(model.multinom)
# 
# # LOOCV
# cvv.multinom <- sapply(X = 1:nrow(newdata2), function(X){
#   training_data <- newdata2[-X,]
#   testing_data <- newdata2[X,]
#   cvmodel <- multinom(sentiment ~ ., data=training_data)
#   guess <- predict(cvmodel, newdata=testing_data)
#   accuracy <- guess == testing_data$sentiment
#   return(accuracy)
# })
# 
# totalaccuracy.multinom <- sum(cvv.multinom) / length(cvv.multinom)
# totalaccuracy.multinom    # 0.5891947

# The LOOCV results show that the multinomial model only predicted sentiment
# correctly about 58.92% of the time. This is worse when compared to the
# first multiple linear regression model and the first ordinal logistic
# regression model that also use all 43 terms as predictor variables. This model
# arguably does worse than a simple prediction of all 3s for sentiments.
# Since this model seems to perform worse than its multiple linear regression
# and ordinal logistic regression counterparts, we decided not to pursue any
# further model development for multinomial models.



# Ultimately, we decided to use the second ordinal logistic regression model
# as our parametric model to predict sentiment for the test data. This is
# because even though its LOOCV predictive accuracy of about 61.47% is lower
# than the 61.88% for the fourth linear regression model, it is not that much
# lower. Also, the ordinal logistic regression model fits the circumstances
# better, conceptually speaking. The multiple linear regression model assumes
# that the response variable sentiment is a numeric variable. In contrast,
# the ordinal logistic regression model more correctly measures sentiment as
# an ordered classification variable. We feel that this better conceptual fit
# justifies the small loss in LOOCV predictive accuracy.



################### Nonparametric K-nearest neighbors ###################

# The goal here was to maximize prediction accuracy using K-Nearest Neighbors
# I found that this was difficult, and I was unable to train a model that performed 
# better than a non-intelligent guess of 3. In other words, my model lacked predictive
# power. My guess is that, as a non-parametric method, KNN lacked the large number of 
# observations required (there were only 1000 observations to train on) to actually 
# buil good predictive power. At the same time, since we're working in text mining with
# term document matrices, there was an issue with the large amount of dimensionality 
# in our training data. However, it wasn't for lack of trying. Over the course of model
# selection I tweaked a large number of possible variables to try to eke out the best
# performance possible. Results are summarized in a plot attached as part of the 
# submission. The LOOCV is implemented as part of the KNN function itself.

# Playing around with weightings
#lapply(X = c('weightTfIdf', 'weightTf', 'weightBin', 'weightSMART'), function(Z){
# I tried all 4 possible weightings to see which one yielded the best results. weightTfIdf was the best
# compute TF-IDF matrix and inspect sparsity
# twitter.tfidf = DocumentTermMatrix(twitter_corpus, control = list(weighting = get(Z)))

# I went through to find the sensitivity to varying sparsity
# most_of_the_way <- bind_rows(lapply(X = seq(.9, .995, .005), function(X){
# I was curious to see what sparsity would produce the best test accuracy. It turns out the closer to 1 you got (more terms), the better
# twitter.tfidf.sparse = twitter.clean.tfidf
# twitter.tfidf.sparse = removeSparseTerms(twitter.clean.tfidf, X)  # remove terms that are absent from at least X% of documents (keep most terms)


# For KNN, we chose to use the TF-IDF matrix with the 99.5% sparsity threshold,
# as that provided us with the best test accuracy when we were comparing various
# sparsity levels between 90% and 99.5%.

# We have a vectorised mode function for quickly calculating the mode
# Taken from tutorialspoint at
# https://www.tutorialspoint.com/r/r_mean_median_mode.htm
getmode <- function(nums) {
  uniqnums <- unique(nums)
  uniqnums[which.max(tabulate(match(nums, uniqnums)))]
}

# Also interested in Cosine Distance. This implementation was found on stackoverflow at 
# https://stackoverflow.com/questions/2535234/find-cosine-similarity-between-two-arrays
cosineDist <- function(x){
  as.dist(1 - x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2))))) 
}

# Now going to code up KNN -- includes LOOCV inside it when test = NA (so when you don't pass a testing set to it)
funcKNN <- function(train, test = NA, prediction_column, k){
  mtrx.tfidf = as.matrix(train)
  # dist.matrix = as.matrix(dist(mtrx.tfidf))   # Euclidean distance
  dist.matrix = as.matrix(cosineDist(mtrx.tfidf)) # Cosine distance ended up producing slightly better accuracy on the test data
  # Some preliminary work follows on the next two lines for trying to reweight counts (this was not successful at all)
  # grpd_sentiment <- group_by(train_data, sentiment)
  # sentiment_percent <- summarise(grpd_sentiment, percent = n() / NROW(grpd_sentiment))
  # Perform cross-validation on the full set of data when test is NA
  if(is.na(test)){
    # Default to LOOCV
    # I also tried cosine distance as a distance metric; it was not particularly useful
    # LOOCV is pretty easy; we don't need to do anything random as the process is deterministic (so no sample function req)
    correct_pred <- sapply(X = 1 : NROW(dist.matrix), function(X){
      # Alright, so the general order here is that the function will go through each row of the distance matrix one at a time
      # and use that row as the test set while all of the other rows are used as the training set
      # Distance matrix
      most.similar.documents = order(dist.matrix[X,], decreasing = FALSE)
      # Use k to find the k-nearest neighbor -- ignore the first one (the test row is closest to itself)
      knn <- most.similar.documents[2:(k+1)]
      # Now pick the value with the most counts
      
      # I did attempt a normalization technique to re-weight the counts of the different sentiments
      # prediction_values <- prediction_column[knn]
      # prediction_value <- which.max(tabulate(match(prediction_values, sentiment_percent$sentiment)) / sentiment_percent$percent)
      
      prediction_value <- getmode(prediction_column[knn])
      # This has you pick the mean of the values (another option itself)
      # prediction_value <- round(mean(prediction_column[knn])) # Old method before when we weren't taking weighting into account
      # return(prediction_value)
      return(prediction_value == prediction_column[X])
    })
    accuracy <- sum(correct_pred) / length(correct_pred)
    accuracy
    # hist(correct_pred) # I wanted to track the distribution of predictions to make sure it wasn't just all 3's
    return(accuracy)
  }
  # Now dealing with the case where we actually want to run the model on some testing set
  else{
    mtrx.test <- as.matrix(test)
    # The next few lines make sure that we only use features from the training set that are also present in the 
    # testing set. o/w you get an error 
    train_column_names <- colnames(mtrx.tfidf)
    test_column_names <- colnames(mtrx.test)
    shared_column_names <- test_column_names[test_column_names %in% train_column_names]
    # Now perform KNN
    sapply(X = 1 : NROW(mtrx.test), function(X){
      # The idea here is that we go through each row of the testing set one at a time and find the distances between it and 
      # every row in the training set. Then we can make a prediction for that test data based on its nearest neighbors.
      train_and_test <- rbind(mtrx.test[X,shared_column_names], mtrx.tfidf[,shared_column_names])
      # Distance matrix
      dist.matrix = as.matrix(cosineDist(train_and_test))
      most.similar.documents = order(dist.matrix[1,], decreasing = FALSE)
      # Use k to find the k-nearest neighbor -- ignore the first one (the test row is closest to itself)
      knn <- most.similar.documents[2:(k+1)]
      # Now pick the value with the most counts
      prediction_value <- getmode(prediction_column[knn])
      return(prediction_value)
    })
  }
}
# k_outputs <- bind_rows(lapply(X = 1:50, function(Y){ # The goal here was to see how sensitive the test accuracy was to varying values of k
#   pred_accuracy <- funcKNN(twitter.tfidf.sparse, prediction_column = train_data$sentiment, k = Y)
#   return(tibble(acc = pred_accuracy, k = Y))
# }))
# k_outputs$scarcity <- X
# return(k_outputs)
# }))
# most_of_the_way$weighting <- Z
# return(most_of_the_way)
# })
# output_kays <-sapply(X = ((1:50) * 3) - 2, function(X){funcKNN(twitter.tfidf.sparse, NA, train_data$sentiment, X)})
# plot(output_kays) # I wanted to visualize how k actually affected the predicted test accuracy

# I ran two versions of funcKNN. One with Euclidean distance and the other with Cosine distance. I wanted to see which produced 
# the better prediction accuracy during CV
# cosine_dist# This is with cosinedist
# cosine_dist$dist <- 'cosine'
# WeightTfIdf is clearly the best
# euclid_dist
# euclid_dist$dist <- 'euclidean'

# survey_of_fits <- bind_rows(cosine_dist, euclid_dist)

# I plotted all of my different CV outcomes when adjusting all of the following levers we could play around with:
# Weighting schemes
# Sparsity filter
# K-values
# Distance measurement -- also angle (cosine similarity)
# The graph is faceted by value of k horizontally and by the distance metric vertically
# The x-axis has the scarcity cutoff
# The y-axis has the predicted accuracy based on LOOCV
# And the color is by weighting used in the term doc matrix
# library(ggplot2)
# ggplot(survey_of_fits, aes(x = scarcity, y = acc, color = weighting)) + 
#   facet_grid(dist~k) + 
#   scale_y_continuous("Accuracy") + 
#   scale_x_continuous("Scarcity cutoff") +
#   geom_point(alpha = .75) +
#   theme_bw()
# The results were not particularly encouraging. You can see immediately that you get a big performance boost from 
# using weightTfIdf as opposed to any other weighting. You also see that the higher k-values produce the larger 
# accuracies when scarcity cutoff is closest to 1. The distance metrics don't seem to be particularly better than one another. 



# However, following the survey of all of all of these cross-validated accuracies, I found that the best accuracy came 
# from k = 9, dist = 'cosine', sparsity = .995, and weighting = 'weightTfIdf'.
# So the KNN function that is run on the test data uses these specified
# parameters, with the corresponding 99.5% TF-IDF matrix.



################### Generate Submissions With Test Data ###################



# Read in the test data
test_data <- read_csv(file = 'test.csv')

# Clean the test data the exact same way as the training data
test_tweets <- as.tibble(test_data$text)
test_corpus <- VCorpus(DataframeSource(test_tweets))
test.tfidf = DocumentTermMatrix(test_corpus, control = list(weighting = weightTfIdf))
test.tfidf
test_corpus.clean = tm_map(test_corpus, stripWhitespace)                          # remove extra whitespace
test_corpus.clean = tm_map(test_corpus.clean, removeNumbers)                      # remove numbers
test_corpus.clean = tm_map(test_corpus.clean, removePunctuation)                  # remove punctuation
test_corpus.clean = tm_map(test_corpus.clean, content_transformer(tolower))       # ignore case
test_corpus.clean = tm_map(test_corpus.clean, removeWords, stopwords("english"))  # remove stop words
test_corpus.clean = tm_map(test_corpus.clean, stemDocument)                       # stem all words
test.clean.tfidf = DocumentTermMatrix(test_corpus.clean, control = list(weighting = weightTfIdf))
test.clean.tfidf

# We see that before cleaning the test data, the TF-IDF matrix shows that there
# are 4,896 distinct terms, 12,682 non-sparse entries, and 4,780,502 sparse
# entries. After cleaning, the TF-IDF matrix gives us 3,308 distinct terms,
# with 10,016 non-sparse entries and 3,228,516 sparse entries. This is similar
# to what we saw with the training data.
# Here we do not need to remove sparse terms as with the training data because
# we are not fitting a model to the test data. Also, removing sparse terms may
# cause predictor terms to be removed from the test data, which is bad. When we
# are making predictions with our models, it is all right to have lots of extra
# variables; the models will just ignore them.


# Parametric submission:
# Using ordinal logistic regression #2 model to predict sentiment on test data

# Format cleaned test data
cleanedtestdata <- as.data.frame(as.matrix(test.clean.tfidf))
newtestdata <- cbind(test_data,cleanedtestdata)

# Generate predictions and format predictions into submittable format
predictions <- as.data.frame(predict(ordinal.model2, newdata=newtestdata))
names(predictions) <- "sentiment"
submission <- cbind(newtestdata$id, predictions)
names(submission) <- c("id","sentiment")

# View summary of predictions
summary(submission)   # 964 3s | 9 4s | 6 5s

# Write predictions out to CSV file
write.csv(submission, "Parametric_Model_Ordinal_Logistic_Regression_Predictions.csv", row.names=FALSE)


# Our parametric model submission did predict mostly 3s, but it also had several
# 4s and 5s within the predictions.
# On submission to Kaggle, we received the following scores for this submission:
# Public score: 0.68507
# Private score: 0.65510



# Nonparametric submission:
# Running KNN on the test data using the training data with k = 9 -- takes a while to run
test.predictions <- funcKNN(twitter.tfidf.sparse, test.clean.tfidf, train_data$sentiment, k = 9)

# Final formatting for submission
submit_this <- test_data
submit_this$sentiment <- test.predictions
submit_this$text <- NULL
submit_this
write_csv(submit_this, 'nonparam_knn.csv')


# On submission to Kaggle, we received the following scores for this submission:
# Public score: 0.65848
# Private score: 0.64693
