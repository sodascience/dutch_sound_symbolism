# This script is used to find the optimal hyperparameter settings for the 
# regression analysis using script 08. The data used in this script is also
# generated using script 08 using the grid search functions.

library(tidyverse)
library(dplyr)

path <- 'processed_data/analyses/grid_search/'

# create a list of all .csv files in the grid search folder
file_list <- list.files(path = path, pattern = '*.csv', full.names = TRUE)

# initialize an empty dataframe to store results in
grid_results <- data.frame()

for (file in file_list){
  # for each file in the file list, read the data
  data <- read.csv(file)
  
  # store the info embedded in the file name
  file_info <- str_split(gsub('processed_data/analyses/grid_search/|_grid-search.csv', '', file), '_')
  
  # using the file name info, store the association, embedding type, and 
  # embedding model/layer
  association <- file_info[[1]][1]
  emb_type <- gsub('[^[:alpha:]]', '', file_info[[1]][2])
  emb_model <- gsub('[[:alpha:]]', '', file_info[[1]][2])
  
  # calculate the average MSE over all grid search folds per nodes-dropout-act-
  # lr-n_layers combination
  grid_result_avg <- data %>%
    dplyr::group_by(nodes, dropout, act, lr, n_layers) %>%
    dplyr::summarise(avg = mean(mse)) %>%
    dplyr::mutate(association = association,
                  emb_type = emb_type,
                  emb_model = emb_model,
                  .before = nodes)
  
  # save the average MSE to the results dataframe
  grid_results <- rbind(grid_results, grid_result_avg)
    
}

# calculate the best performing hyperparameter settings (i.e., lowest MSE) 
# per association-emb_type-emb_model-n_layers combination
best_models <- grid_results %>%
  dplyr::group_by(association, emb_type, emb_model, n_layers) %>%
  dplyr::slice_min(avg)

# save the dataframe with the best performing hyperparameter settings to be used
# for the regression analysis
write.csv(best_models, file = 'results/grid_search/best_hyperparameters.csv', row.names = FALSE)