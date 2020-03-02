library(XML)
library(tm)
library(tidyverse)


test.data <- read.csv("test.csv", stringsAsFactors = FALSE) 
test.df<-as.data.frame(test.data$text, stringsAsFactors = FALSE)


#sentiment <- sample(1:5, 979, replace = T)
#pred1 <- as.data.frame(sentiment)
#write.csv(pred1,"kaggle_twitter_yh8a_submission1.csv")


news = VCorpus(DataframeSource(test.df))

news.tfidf = DocumentTermMatrix(news, control = list(weighting = weightTfIdf))
news.tfidf
news.tfidf[1:5,1:5]
as.matrix(news.tfidf[1:5,1:5])

# there's a lot in the documents that we don't care about. clean up the corpus.
news.clean = tm_map(news, stripWhitespace)                          # remove extra whitespace
news.clean = tm_map(news.clean, removeNumbers)                      # remove numbers
news.clean = tm_map(news.clean, removePunctuation)                  # remove punctuation
news.clean = tm_map(news.clean, content_transformer(tolower))       # ignore case
news.clean = tm_map(news.clean, removeWords, stopwords("english"))  # remove stop words
news.clean = tm_map(news.clean, stemDocument)                       # stem all words

# compare original content of document 1 with cleaned content
news[[1]]$content
news.clean[[1]]$content  # do we care about misspellings resulting from stemming?

# recompute TF-IDF matrix
news.clean.tfidf = DocumentTermMatrix(news.clean, control = list(weighting = weightTfIdf))
news.clean.tfidf

# reinspect the first 5 documents and first 5 terms
news.clean.tfidf[1:5,1:5]
as.matrix(news.clean.tfidf[1:5,1:5])

# we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
tfidf.99 = removeSparseTerms(news.clean.tfidf, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
tfidf.99
as.matrix(tfidf.99[1:5,1:5])
as.matrix(tfidf.99)
#as.data.frame(tfidf.99)
cleanedtest99 <- as.data.frame(as.matrix(tfidf.99))

tfidf.98 = removeSparseTerms(news.clean.tfidf, 0.98)  # remove terms that are absent from at least 99% of documents (keep most terms)
tfidf.98
as.matrix(tfidf.98[1:5,1:5])
as.matrix(tfidf.98)
cleanedtest98 <- as.data.frame(as.matrix(tfidf.98))

#cleanedtrain <- as.data.frame(as.matrix(tfidf.98))
#sentiment <- train.data$sentiment
#new.train <- cbind(sentiment,cleanedtrain)

#newtest <- cbind(data,cleanedtest)
#as.data.frame(tfidf.98)

# Removes too many
# tfidf.70 = removeSparseTerms(news.clean.tfidf, 0.70)  # remove terms that are absent from at least 70% of documents
# tfidf.70
# as.matrix(tfidf.70[1:5, 1:5])
# news.clean[[1]]$content

# # which documents are most similar?
# dtm.tfidf.99 = as.matrix(tfidf.99)
# dtm.dist.matrix = as.matrix(dist(dtm.tfidf.99))
# most.similar.documents = order(dtm.dist.matrix[1,], decreasing = FALSE)
# news[[most.similar.documents[1]]]$content
# news[[most.similar.documents[2]]]$content
# news[[most.similar.documents[3]]]$content
# news[[most.similar.documents[4]]]$content
# news[[most.similar.documents[5]]]$content


train.lg <- lm(sentiment~., data=new.train)
summary(train.lg)

names(cleanedtest)
new.test <- cleanedtest[,-going]

pred <- data.frame(predict(train.lm99, newdata = cleanedtest))
names(pred) <- c("sentiment")


