# This script is used to perform statistical analyses using the cosine score 
# data generated using the previous script 06

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
survey_ratings <- readRDS('processed_data/survey_ratings/word_scores.rds')

# change the direction of the 'slecht' ratings since the cosine scores measure 
#'good'-ness, and lowercast the word column
survey_ratings$mean[survey_ratings$association == 'slecht'] <- as.numeric(-survey_ratings$mean[survey_ratings$association == 'slecht'])
survey_ratings$word <- tolower(survey_ratings$word)

# create a mapping of words to be changed in the survey_ratings data to make it
# more compatible with the cosine scores data
word_map <- c('vrouwelijk' = 'feminine',
              'slecht' = 'good',            # I understand this is not a translation, direction
              'slim' = 'smart',             # of mean rating will be changed accordingly
              'betrouwbaar' = 'trustworthy',
              'bedrijfsnamen' = 'company',
              'namen' = 'fnames',
              'nepwoorden' = 'nonword')

survey_ratings[, 11:13] <- mutate_all(survey_ratings[, 11:13], ~ str_replace_all(., word_map))

# rename the word_type column 
colnames(survey_ratings)[13] <- 'word_type'

# create DF subsets for all four associations
fem_ratings <- subset(survey_ratings, association == 'feminine', select = c('mean', 'word', 'association', 'word_type'))
good_ratings <- subset(survey_ratings, association == 'good', select = c('mean', 'word', 'association', 'word_type'))
smart_ratings <- subset(survey_ratings, association == 'smart', select = c('mean', 'word', 'association', 'word_type'))
trust_ratings <- subset(survey_ratings, association == 'trustworthy', select = c('mean', 'word', 'association', 'word_type'))


##  CORRELATION ANALYSIS
# initialize some lists to create for-loops with
ratings_list = list(fem_ratings, good_ratings, smart_ratings, trust_ratings)
associations <- list('feminine', 'good', 'smart', 'trustworthy')
word_types <- list('company', 'fnames', 'nonword')

# FastText
cosines_list_ft = list(fem_ft, good_ft, smart_ft, trust_ft)
ft_models <- list('0', '2-5') 

# initialize an empty dataframe to store results in
correlations_ft = data.frame(matrix(ncol = 5, nrow = 0))
colnames(correlations_ft) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (i in 1:length(associations)) {
  for (wordtype in word_types) {
    for (modeltype in ft_models){
      # for every association-word_type-ft_model combination, fetch the relevant 
      # survey rating data and cosine data
      association <- associations[i]
      
      temp_ratings <- subset(data.frame(ratings_list[i]), word_type == wordtype)
      
      temp_data <- data.frame(cosines_list_ft[i])
      temp_data <- unique(subset(temp_data, 
                                 word_type == wordtype & model == modeltype, 
                                 select = c('name', 'delta_all_names')))
      
      # merge the two dataframes
      cor_df <- merge(temp_ratings, temp_data, by.x = 'word', by.y = 'name')
      
      # perform correlation analysis on the survey ratings and cosine scores
      corr <- cor(cor_df$mean, 
                  cor_df$delta_all_names)
      
      # save the correlation for this association-word_type-ft_model combination
      # in the results dataframe
      correlations_ft[nrow(correlations_ft)+1, ] <- c('ft', association, wordtype, modeltype, corr)
    
    }
  }
}  

# save the correlation results as a .csv file
write.csv(correlations_ft, 
          file = paste(path_results, 'correlations_ft0_&_2-5_bootstrap=False.csv', sep = ''), 
          row.names = FALSE)


# BERT
cosines_list_bert = list(fem_bert, good_bert, smart_bert, trust_bert)
bert_models <- list('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11') 

# initialize an empty dataframe to store results in
correlations_bert = data.frame(matrix(ncol = 5, nrow = 0))
colnames(correlations_bert) <- c('emb_type', 'association', 'word_type', 'model', 'correlation')

for (i in 1:length(associations)) {
  for (wordtype in word_types) {
    for (modeltype in bert_models){
      # for every association-word_type-bert_model combination, fetch the relevant 
      # survey rating data and cosine data
      association <- associations[i]
      
      temp_ratings <- subset(data.frame(ratings_list[i]), word_type == wordtype)
      
      temp_data <- data.frame(cosines_list_bert[i])
      temp_data <- unique(subset(temp_data, 
                                 word_type == wordtype & model == modeltype, 
                                 select = c('name', 'delta_all_names')))
      
      # merge the two dataframes
      cor_df <- merge(temp_ratings, temp_data, by.x = 'word', by.y = 'name')
      
      # perform correlation analysis on the survey ratings and cosine scores
      corr <- cor(cor_df$mean, 
                  cor_df$delta_all_names)
      
      # save the correlation for this association-word_type-ft_model combination
      # in the results dataframe
      correlations_bert[nrow(correlations_bert)+1, ] <- c('bert', association, wordtype, modeltype, corr)
      
    }
  }
}  

# save the correlation results as a .csv file
write.csv(correlations_bert, 
          file = paste(path_results, 'correlations_bert-all-layers_bootstrap=False.csv', sep = ''), 
          row.names = FALSE)


## EXTRA PROCESSING OF DATA
# join the fasttext and bert correlation results together into one dataframe
correlations_all <- full_join(correlations_ft, correlations_bert)

# save the combined correlation results as a .csv file
write.csv(correlations_all, 
          file = paste(path_results, 'correlations_ft0&2-5_&_bert-all-layers_bootstrap=False.csv', sep = ''),
          row.names = FALSE)

# find the highest correlation score per embedding type (bert/fasttext) - 
# association - word type combination 
correlations_highest <- correlations_all %>%
  dplyr::group_by(emb_type, association, word_type) %>%
  dplyr::slice_max(correlation)

# create a unified fastText and BERT cosine score dataframe, add and rename some
# columns for better interpretability, factorize columns with text data, and 
# save it as a .rds file
similarity_delta_df <-
  bind_rows(
    fasttext = bind_rows(cosines_list_ft),
    bert = bind_rows(cosines_list_bert) %>%
      mutate(model = as.character(model)),
    .id = "model_type"
  ) %>%
  rename(similarity_delta = delta_all_names) %>%
  mutate(across(where(is.character), as_factor)) 

write_rds(similarity_delta_df, file=paste0(path_results, 'similarity_delta_df.rds'))
