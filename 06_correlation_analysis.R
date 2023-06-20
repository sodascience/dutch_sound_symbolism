library(readr)
library(stringr)
library(dplyr)
library(randomForest)

# So, I'm going to change this file quite radically, so I'm writing down what I 
# will need to do.

# First, I need to write some code in which I analyze the correlation between
# the delta_all_names for each word_type-association-model combination and the
# survey data ratings (word norms), and then save this as a nice and understandable
# table

# Second, I need to write some code in which I probe the bootstrap sample delta's
# against the delta_all_names and delta_survey_poles. I don't know exactly how I 
# will tackle this problem, but it's a type of robustness check so it is important



## LOAD DATA
# Cosine Score DFs
path <- 'processed_data/analyses/correlation_analysis/'

fem_ft <- read_csv(paste(path, 'feminine_ft0_&_2-5_cosine_bootstrap_scores.csv', sep = ''))
good_ft <- read_csv(paste(path, 'good_ft0_&_2-5_cosine_bootstrap_scores.csv', sep = ''))
smart_ft <- read_csv(paste(path, 'smart_ft0_&_2-5_cosine_bootstrap_scores.csv', sep = ''))
trust_ft <- read_csv(paste(path, 'trustworthy_ft0_&_2-5_cosine_bootstrap_scores.csv', sep = ''))

# Survey Ratings
word_map <- c('vrouwelijk' = 'feminine',
              'slecht' = 'good',            # I understand this is not a translation, direction
              'slim' = 'smart',             # of mean rating will be changed accordingly
              'betrouwbaar' = 'trustworthy',
              'bedrijfsnamen' = 'company',
              'namen' = 'fnames',
              'nepwoorden' = 'nonword')

survey_ratings <- readRDS('processed_data/survey_ratings/word_scores.rds')
survey_ratings$mean[survey_ratings$association == 'slecht'] <- as.numeric(-survey_ratings$mean[survey_ratings$association == 'slecht'])
survey_ratings$word <- tolower(survey_ratings$word)
survey_ratings[, 11:13] <- mutate_all(survey_ratings[, 11:13], ~ str_replace_all(., word_map))
colnames(survey_ratings)[13] <- 'word_type'

fem_ratings <- subset(survey_ratings, association == 'feminine', select = c('mean', 'word', 'association', 'word_type'))
good_ratings <- subset(survey_ratings, association == 'good', select = c('mean', 'word', 'association', 'word_type'))
smart_ratings <- subset(survey_ratings, association == 'smart', select = c('mean', 'word', 'association', 'word_type'))
trust_ratings <- subset(survey_ratings, association == 'trustworthy', select = c('mean', 'word', 'association', 'word_type'))



##  CORRELATION ANALYSIS
ratings_list = list(fem_ratings, good_ratings, smart_ratings, trust_ratings)
cosines_list = list(fem_ft, good_ft, smart_ft, trust_ft)

ft_models <- list('0', '2-5') 
associations <- list('feminine', 'good', 'smart', 'trustworthy')
word_types <- list('company', 'fnames', 'nonword')

correlations = data.frame(matrix(ncol = 5, nrow = 0))
colnames(correlations) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (i in 1:length(associations)) {
  for (wordtype in word_types) {
    for (modeltype in ft_models){
      association <- associations[i]
      
      temp_ratings <- subset(data.frame(ratings_list[i]), word_type == wordtype)
      
      temp_data <- data.frame(cosines_list[i])
      temp_data <- unique(subset(temp_data, 
                                 word_type == wordtype & model == modeltype, 
                                 select = c('name', 'delta_all_names')))
    
      cor_df <- merge(temp_ratings, temp_data, by.x = 'word', by.y = 'name')
      
      corr <- cor(temp_ratings$mean, 
                  temp_data$delta_all_names)
      
      correlations[nrow(correlations)+1, ] <- c('ft', association, wordtype, modeltype, corr)
    
    }
  }
}  
  


## FEATURE IMPORTANCE (ROBUSTNESS CHECK)
# FastText

# Feminine
df_fem_fi_ft <- fem_ft[, c(5, 8:ncol(fem_ft))]
df_fem_fi_ft_lex <- subset(fem_ft, model == '0', select = c(5, 8:ncol(fem_ft)))
df_fem_fi_ft_ngram <- subset(fem_ft, model == '2-5', select = c(5, 8:ncol(fem_ft)))

fem_ft_fi_model <- randomForest(delta ~ ., 
                                data = df_fem_fi_ft,
                                ntree = 1000,
                                importance = TRUE)
summary(fem_ft_fi_model)
fem_ft_fi_model$importance


fem_ft_fi_model_lex <- randomForest(delta ~ ., 
                                    data = df_fem_fi_ft_lex,
                                    ntree = 2000,
                                    importance = TRUE)
summary(fem_ft_fi_model_lex)
fem_ft_fi_model_lex$importance


fem_ft_fi_model_ngram <- randomForest(delta ~ ., 
                                      data = df_fem_fi_ft_ngram,
                                      ntree = 2000,
                                      importance = TRUE)
summary(fem_ft_fi_model_ngram)
fem_ft_fi_model_ngram$importance

# Good
df_good_fi_ft <- good_ft[, c(5, 8:ncol(good_ft))]
df_good_fi_ft_ngram <- subset(good_ft, model == '2-5', select = c(5, 8:ncol(good_ft)))

good_ft_fi_model_ngram <- randomForest(delta ~ ., 
                                       data = df_good_fi_ft_ngram,
                                       ntree = 2000,
                                       importance = TRUE)
summary(good_ft_fi_model_ngram)
good_ft_fi_model_ngram$importance

# Smart
df_smart_fi_ft <- smart_ft[, c(5, 8:ncol(smart_ft))]
df_smart_fi_ft_ngram <- subset(smart_ft, model == '2-5', select = c(5, 8:ncol(smart_ft)))

smart_ft_fi_model_ngram <- randomForest(delta ~ ., 
                                        data = df_smart_fi_ft_ngram,
                                        ntree = 2000,
                                        importance = TRUE)
summary(smart_ft_fi_model_ngram)
smart_ft_fi_model_ngram$importance

# Trustworthy
df_trust_fi_ft <- trust_ft[, c(5, 8:ncol(trust_ft))]
df_trust_fi_ft_ngram <- subset(trust_ft, model == '2-5', select = c(5, 8:ncol(trust_ft)))

trust_ft_fi_model_ngram <- randomForest(delta ~ ., 
                                        data = df_trust_fi_ft_ngram,
                                        ntree = 2000,
                                        importance = TRUE)
summary(trust_ft_fi_model_ngram)
trust_ft_fi_model_ngram$importance

# BERT

df_fem_fi_bert <- 

df_good_fi_bert <- 

df_smart_fi_bert <- 

df_trust_fi_bert <- 
  