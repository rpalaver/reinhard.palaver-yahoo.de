# setup: required packages
library(tidyverse)
library(caret)
library(data.table)
library(dslabs)
library(dplyr)
library(gridExtra)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# show edx dataset at the report
edx %>% slice(1:10) %>% knitr::kable()

# show names from dataset at the report
names(edx)

# show number of different rating at the report
edx %>% group_by(rating) %>% summarize(number=n())

# compute number of rows
dim(edx)[1]

# compute number of columns
dim(edx)[2]

# compute number of movies
n_distinct(edx$movieId)

# compute number of users
n_distinct(edx$userId)

# create an image for a random sample of 100 movies and 100 users
users <- sample(unique(edx$userId), 100)
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")
rm(users)

# create distribution plots for movies and users  
p1 <- edx %>% # create plot for distribution of ratings per movie
  dplyr::count(movieId) %>% arrange(desc(n)) %>%
  ggplot(aes(n)) + 
  geom_histogram(fill = "cadetblue",bins = 20, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

p2 <- edx %>% # create plot for distribution of ratings per user
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(fill = "cadetblue",bins = 20, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")
grid.arrange(p1,p2,ncol = 2) # arrange the plots into a grid
rm(p1,p2)

# create test and train sets (test set will be 20% of edx data)
set.seed(755, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[test_index,]
test_set <- edx[-test_index,]
rm(test_index)

# check all users and movies exists in both datasets
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# create RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# average of all ratings
mu <- mean(train_set$rating)
mu

# first modell 'naive RMSE'
naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse

# create table for differenz approaches
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

# show varibility of movie effects
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
qplot(b_i, data = movie_avgs, bins = 30, color = I("black")) + ggtitle("Movie effect")

# model with movie effects
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse))

options(digits = 2)
rmse_results %>% knitr::kable()

# show varibility of user effects
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = I("black")) + ggtitle("User effect")

# compute b_u with user and movie effects
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# model with user and movie effects
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="User/movie Effect Model",
                                     RMSE = model_2_rmse))
options(digits = 3)
rmse_results %>% knitr::kable()

# 10 largest mistakes only considering movie effects
test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) %>% knitr::kable()

# create database with movie titles
movie_titles <- train_set %>% 
  select(movieId, title) %>%
  distinct()

# show best 10 movies
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

# show worst 10 movies
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

# show how often the top 10 are rated
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>%
  knitr::kable()

# show how often the lowest 10 are rated  
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# estimate movie effects with lambda 3
lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

# plot original versus regularlized b_i
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# top 10 movies with regularlized b_i
train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# worst 10 movies with regularlized b_i
train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# estimate RMSE with regularlized averages and movie effects
predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

# cross validation with different lambdas for movie effect
lambdas <- seq(0, 10, 0.25)
mu <- mean(train_set$rating)
just_the_sum <- train_set %>%
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)

# show optimized lambda
lambdas[which.min(rmses)]

# cross validation with different lambdas for movie and user effect
lambdas <- seq(1, 6, 0.05)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)


# show optimized lambda
lambdas[which.min(rmses)]

# estimate RMSE with regularlized averages with movie and user effects
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

# final model with edx and validation datasets
lambda <- lambdas[which.min(rmses)]

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
final_rmse <-RMSE(predicted_ratings, validation$rating)


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="final model with datasets edx vs. validation",  
                                     RMSE = final_rmse))
options(digits = 6)
rmse_results %>% knitr::kable()
