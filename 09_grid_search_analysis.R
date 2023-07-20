library(tidyverse)
library(dplyr)

path <- 'processed_data/analyses/grid_search/'

file_list <- list.files(path = path, pattern = '*.csv', full.names = TRUE)

grid_results <- data.frame()

for (file in file_list){
  data <- read.csv(file)
  
  file_info <- str_split(gsub('processed_data/analyses/grid_search/|_grid-search.csv', '', file), '_')
  
  association <- file_info[[1]][1]
  emb_type <- gsub('[^[:alpha:]]', '', file_info[[1]][2])
  emb_model <- gsub('[[:alpha:]]', '', file_info[[1]][2])
  
  grid_result_avg <- data %>%
    dplyr::group_by(nodes, dropout, act, lr, n_layers) %>%
    dplyr::summarise(avg = mean(mse)) %>%
    dplyr::mutate(association = association,
                  emb_type = emb_type,
                  emb_model = emb_model,
                  .before = nodes)
  
  grid_results <- rbind(grid_results, grid_result_avg)
    
}

best_models <- grid_results %>%
  dplyr::group_by(association, emb_type, emb_model, n_layers) %>%
  dplyr::slice_min(avg)

write.csv(best_models, file = 'results/grid_search/best_hyperparameters.csv', row.names = FALSE)