library(readr)
library(stringr)
library(dplyr)
library(randomForest)

## LOAD DATA
# Cosine Score DFs
path <- 'processed_data/analyses/correlation_analysis/'
path_results <- 'results/analyses/correlation_analysis/'

fem_ft <- read_csv(paste(path, 'feminine_ft0_&_2-5_bootstrap=False_cosine_scores.csv', sep = ''))
good_ft <- read_csv(paste(path, 'good_ft0_&_2-5_bootstrap=False_cosine_scores.csv', sep = ''))
smart_ft <- read_csv(paste(path, 'smart_ft0_&_2-5_bootstrap=False_cosine_scores.csv', sep = ''))
trust_ft <- read_csv(paste(path, 'trustworthy_ft0_&_2-5_bootstrap=False_cosine_scores.csv', sep = ''))

fem_bert <- read_csv(paste(path, 'feminine_bert-all-layers_bootstrap=False_cosine_scores.csv', sep = ''))
good_bert <- read_csv(paste(path, 'good_bert-all-layers_bootstrap=False_cosine_scores.csv', sep = ''))
smart_bert <- read_csv(paste(path, 'smart_bert-all-layers_bootstrap=False_cosine_scores.csv', sep = ''))
trust_bert <- read_csv(paste(path, 'trustworthy_bert-all-layers_bootstrap=False_cosine_scores.csv', sep = ''))

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
associations <- list('feminine', 'good', 'smart', 'trustworthy')
word_types <- list('company', 'fnames', 'nonword')

# FastText

cosines_list_ft = list(fem_ft, good_ft, smart_ft, trust_ft)
ft_models <- list('0', '2-5') 
correlations_ft = data.frame(matrix(ncol = 5, nrow = 0))
colnames(correlations_ft) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (i in 1:length(associations)) {
  for (wordtype in word_types) {
    for (modeltype in ft_models){
      association <- associations[i]
      
      temp_ratings <- subset(data.frame(ratings_list[i]), word_type == wordtype)
      
      temp_data <- data.frame(cosines_list_ft[i])
      temp_data <- unique(subset(temp_data, 
                                 word_type == wordtype & model == modeltype, 
                                 select = c('name', 'delta_all_names')))
    
      cor_df <- merge(temp_ratings, temp_data, by.x = 'word', by.y = 'name')
      
      corr <- cor(temp_ratings$mean, 
                  temp_data$delta_all_names)
      
      correlations_ft[nrow(correlations_ft)+1, ] <- c('ft', association, wordtype, modeltype, corr)
    
    }
  }
}  

write.csv(correlations_ft, 
          file = paste(path_results, 'correlations_ft0_&_2-5_bootstrap=False.csv', sep = ''), 
          row.names = FALSE)


# BERT
cosines_list_bert = list(fem_bert, good_bert, smart_bert, trust_bert)
bert_models <- list('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11') 
correlations_bert = data.frame(matrix(ncol = 5, nrow = 0))
colnames(correlations_bert) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (i in 1:length(associations)) {
  for (wordtype in word_types) {
    for (modeltype in bert_models){
      association <- associations[i]
      
      temp_ratings <- subset(data.frame(ratings_list[i]), word_type == wordtype)
      
      temp_data <- data.frame(cosines_list_bert[i])
      temp_data <- unique(subset(temp_data, 
                                 word_type == wordtype & model == modeltype, 
                                 select = c('name', 'delta_all_names')))
      
      cor_df <- merge(temp_ratings, temp_data, by.x = 'word', by.y = 'name')
      
      corr <- cor(temp_ratings$mean, 
                  temp_data$delta_all_names)
      
      correlations_bert[nrow(correlations_bert)+1, ] <- c('bert', association, wordtype, modeltype, corr)
      
    }
  }
}  

write.csv(correlations_bert, 
          file = paste(path_results, 'correlations_bert-all-layers_bootstrap=False.csv', sep = ''), 
          row.names = FALSE)

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
write.csv(fem_ft_fi_model$importance,
          file = paste(path_results, 'fem_ft_cor_dt_feat-imp_scores.csv', sep = ''),
          row.names = TRUE)


fem_ft_fi_model_lex <- randomForest(delta ~ ., 
                                    data = df_fem_fi_ft_lex,
                                    ntree = 1000,
                                    importance = TRUE)
summary(fem_ft_fi_model_lex)
fem_ft_fi_model_lex$importance
write.csv(fem_ft_fi_model_lex$importance,
          file = paste(path_results, 'fem_ft-lexical_cor_dt_feat-imp_scores.csv', sep = ''),
          row.names = TRUE)


fem_ft_fi_model_ngram <- randomForest(delta ~ ., 
                                      data = df_fem_fi_ft_ngram,
                                      ntree = 1000,
                                      importance = TRUE)
summary(fem_ft_fi_model_ngram)
fem_ft_fi_model_ngram$importance
write.csv(fem_ft_fi_model_ngram$importance,
          file = paste(path_results, 'fem_ft2-5_cor_dt_feat-imp_scores.csv', sep = ''),
          row.names = TRUE)

# Good
df_good_fi_ft <- good_ft[, c(5, 8:ncol(good_ft))]
df_good_fi_ft_ngram <- subset(good_ft, model == '2-5', select = c(5, 8:ncol(good_ft)))

good_ft_fi_model_ngram <- randomForest(delta ~ ., 
                                       data = df_good_fi_ft_ngram,
                                       ntree = 1000,
                                       importance = TRUE)
summary(good_ft_fi_model_ngram)
good_ft_fi_model_ngram$importance
write.csv(good_ft_fi_model_ngram$importance,
          file = paste(path_results, 'good_ft2-5_cor_dt_feat-imp_scores.csv', sep = ''),
          row.names = TRUE)

# Smart
df_smart_fi_ft <- smart_ft[, c(5, 8:ncol(smart_ft))]
df_smart_fi_ft_ngram <- subset(smart_ft, model == '2-5', select = c(5, 8:ncol(smart_ft)))

smart_ft_fi_model_ngram <- randomForest(delta ~ ., 
                                        data = df_smart_fi_ft_ngram,
                                        ntree = 1000,
                                        importance = TRUE)
summary(smart_ft_fi_model_ngram)
smart_ft_fi_model_ngram$importance
write.csv(smart_ft_fi_model_ngram$importance,
          file = paste(path_results, 'smart_ft2-5_cor_dt_feat-imp_scores.csv', sep = ''),
          row.names = TRUE)

# Trustworthy
df_trust_fi_ft <- trust_ft[, c(5, 8:ncol(trust_ft))]
df_trust_fi_ft_ngram <- subset(trust_ft, model == '2-5', select = c(5, 8:ncol(trust_ft)))

trust_ft_fi_model_ngram <- randomForest(delta ~ ., 
                                        data = df_trust_fi_ft_ngram,
                                        ntree = 1000,
                                        importance = TRUE)
summary(trust_ft_fi_model_ngram)
trust_ft_fi_model_ngram$importance
write.csv(trust_ft_fi_model_ngram$importance,
          file = paste(path_results, 'trust_ft2-5_cor_dt_feat-imp_scores.csv', sep = ''),
          row.names = TRUE)

# BERT

# Feminine
df_fem_fi_bert <- fem_bert[, c(5, 8:ncol(fem_bert))]

fem_bert_fi_model <- randomForest(delta ~ ., 
                                data = df_fem_fi_bert,
                                ntree = 100,
                                importance = TRUE)
summary(fem_bert_fi_model)
fem_bert_fi_model$importance
write.csv(fem_bert_fi_model$importance,
          file = paste(path_results, 'fem_bert-all_cor_dt_feat-imp_scores.csv', sep = ''),
          row.names = TRUE)

# Good
df_good_fi_bert <- good_bert[, c(5, 8:ncol(good_bert))]

good_bert_fi_model <- randomForest(delta ~ ., 
                                       data = df_good_fi_bert,
                                       ntree = 100,
                                       importance = TRUE)
summary(good_bert_fi_model)
good_bert_fi_model$importance
write.csv(good_bert_fi_model$importance,
          file = paste(path_results, 'good_bert-all_cor_dt_feat-imp_scores.csv', sep = ''),
          row.names = TRUE)

# Smart
df_smart_fi_bert <- smart_bert[, c(5, 8:ncol(smart_bert))]

smart_bert_fi_model <- randomForest(delta ~ ., 
                                        data = df_smart_fi_bert,
                                        ntree = 100,
                                        importance = TRUE)
summary(smart_bert_fi_model)
smart_bert_fi_model$importance
write.csv(smart_bert_fi_model$importance,
          file = paste(path_results, 'smart_bert-all_cor_dt_feat-imp_scores.csv', sep = ''),
          row.names = TRUE)

# Trustworthy
df_trust_fi_bert <- trust_bert[, c(5, 8:ncol(trust_bert))]

trust_bert_fi_model <- randomForest(delta ~ ., 
                                        data = df_trust_fi_bert,
                                        ntree = 100,
                                        importance = TRUE)
summary(trust_bert_fi_model)
trust_bert_fi_model$importance
write.csv(trust_bert_fi_model$importance,
          file = paste(path_results, 'trust_bert-all_cor_dt_feat-imp_scores.csv', sep = ''),
          row.names = TRUE)

