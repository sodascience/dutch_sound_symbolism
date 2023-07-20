library(readr)
library(stringr)
library(dplyr)
library(randomForest)

## LOAD DATA
# Cosine Score DFs
path <- 'processed_data/analyses/correlation_analysis/'
path_results <- 'results/analyses/correlation_analysis/'

fem_ft <- read_csv(paste0(path, 'gender_ft0_&_2-5_bootstrap=False_cosine_scores.csv'))
fem_bert <- read_csv(paste0(path, 'gender_bert-all-layers_bootstrap=False_cosine_scores.csv'))

con_ft <- read_csv(paste0(path, 'concreteness_ft0_&_2-5_bootstrap=False_cosine_scores.csv'))
con_bert <- read_csv(paste0(path, 'concreteness_bert-all-layers_bootstrap=False_cosine_scores.csv'))

# Survey Ratings
fem_ratings <- read_rds('processed_data/vankrunkelsven_2022.rds')
  
con_ratings <- read_rds('processed_data/brysbaert_2014.rds')

## Concreteness
# FastText
wordtype <- 'real_word'
association <- 'concreteness'
ft_models <- list('0', '2-5') 
con_correlations_ft = data.frame(matrix(ncol = 5, nrow = 0))
colnames(con_correlations_ft) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (modeltype in ft_models){
  temp_data <- unique(subset(con_ft, 
                             word_type == wordtype & model == modeltype, 
                             select = c('name', 'delta_all_names')))
  
  cor_df <- merge(con_ratings, temp_data, by.x = 'word', by.y = 'name')
  
  corr <- cor(cor_df$concreteness_brysbaert, 
              cor_df$delta_all_names,
              use = 'complete.obs')
  
  con_correlations_ft[nrow(con_correlations_ft)+1, ] <- c('ft', association, wordtype, modeltype, corr)
}

# BERT
wordtype <- 'real_word'
association <- 'concreteness'
bert_models <- list('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',  '11') 
con_correlations_bert = data.frame(matrix(ncol = 5, nrow = 0))
colnames(con_correlations_bert) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (modeltype in bert_models){
  temp_data <- unique(subset(con_bert, 
                             word_type == wordtype & model == modeltype, 
                             select = c('name', 'delta_all_names')))
  
  cor_df <- merge(con_ratings, temp_data, by.x = 'word', by.y = 'name')
  
  corr <- cor(cor_df$concreteness_brysbaert, 
              cor_df$delta_all_names,
              use = 'complete.obs')
  
  con_correlations_bert[nrow(con_correlations_bert)+1, ] <- c('bert', association, wordtype, modeltype, corr)
}

## Gender
# FastText
wordtype <- 'real_word'
association <- 'feminine'
ft_models <- list('0', '2-5') 
fem_correlations_ft = data.frame(matrix(ncol = 5, nrow = 0))
colnames(fem_correlations_ft) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (modeltype in ft_models){
  temp_data <- unique(subset(fem_ft, 
                             word_type == wordtype & model == modeltype, 
                             select = c('name', 'delta_all_names')))
  
  cor_df <- merge(fem_ratings, temp_data, by.x = 'word', by.y = 'name')
  
  corr <- cor(cor_df$gender_vankrunkelsven, 
              cor_df$delta_all_names,
              use = 'complete.obs')
  
  fem_correlations_ft[nrow(fem_correlations_ft)+1, ] <- c('ft', association, wordtype, modeltype, corr)
}

# BERT
wordtype <- 'real_word'
association <- 'feminine'
bert_models <- list('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',  '11') 
fem_correlations_bert = data.frame(matrix(ncol = 5, nrow = 0))
colnames(fem_correlations_bert) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (modeltype in bert_models){
  temp_data <- unique(subset(fem_bert, 
                             word_type == wordtype & model == modeltype, 
                             select = c('name', 'delta_all_names')))
  
  cor_df <- merge(fem_ratings, temp_data, by.x = 'word', by.y = 'name')
  
  corr <- cor(cor_df$gender_vankrunkelsven, 
              cor_df$delta_all_names,
              use = 'complete.obs')
  
  fem_correlations_bert[nrow(fem_correlations_bert)+1, ] <- c('bert', association, wordtype, modeltype, corr)
}

