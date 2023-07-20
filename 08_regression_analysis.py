'''
This script is used to generate the data to be used for the regression analysis, which is performed 
with script 09. 

In this script the data generated under script 05 is used. First, grid search is performed to find 
the optimal hyperparameter settings for the regression analysis. Then, the regression analysis is 
performed and dataframes with regression scores are automatically saved.
'''

from src.analyses.regression_analysis import grids_searcher, open_processed_wordscores_rds, regression_analysis
import pickle
import pandas as pd

## LOAD DATA
with open('./processed_data/analyses/dataframes/survey_data_ft_df.pkl', 'rb') as f:
    survey_ft_df = pickle.load(f)   #['name', 'word_type', 'embedding_type', 'model', 'embedding']

with open('./processed_data/analyses/dataframes/survey_data_bert_df.pkl', 'rb') as f:
    survey_bert_df = pickle.load(f)

# load the survey rating scores
word_scores = open_processed_wordscores_rds()

## SET IMPORTANT PARAMETERS
LEXICAL = False
M = 2
RND_ITERS = 50
FEATURE = 'embedding'
TARGET = 'mean_rating'
GROUP = 'word_type'
ITEM = 'name'
FT_MODELS = ['0', '2-5']
BERT_LAYERS = list(range(12))
ASSOCIATIONS = ['feminine', 'good', 'smart', 'trustworthy']


## GRID SEARCH
# perform grid search to find the optimal hyperparameter settings per association

# FASTTEXT
# merge the survey ratings with the fasttext embedding data
merged_ft_df = pd.merge(survey_ft_df[survey_ft_df['model'].isin(FT_MODELS)], 
                        word_scores, 
                        on=['name', 'word_type'], 
                        how='inner').reindex(
                        columns = ['name', 'word_type', 'association', 'mean_rating', 'embedding_type', 'model', 'embedding'])

# perform grid search for all associations
grids_searcher(df = merged_ft_df,
               associations = ASSOCIATIONS, 
               x_col = FEATURE, 
               y_col = TARGET, 
               group_col = GROUP, 
               emb_model = FT_MODELS)

# BERT
# merge the survey ratings with the BERT embedding data
merged_bert_df = pd.merge(survey_bert_df[survey_bert_df['model'].isin(BERT_LAYERS)], 
                          word_scores, 
                          on=['name', 'word_type'], 
                          how='inner').reindex(
                          columns = ['name', 'word_type', 'association', 'mean_rating', 'embedding_type', 'model', 'embedding'])

# perform grid search for all associations
grids_searcher(df = merged_bert_df,
               associations = ASSOCIATIONS, 
               x_col = FEATURE, 
               y_col = TARGET, 
               group_col = GROUP, 
               emb_model = BERT_LAYERS)



## REGRESSION ANALYSIS
# load the optimal hyperparameter settings
HYPERPARAMETERS = pd.read_csv('./results/grid_search/best_hyperparameters_first-names-lexical_True.csv')

# FASTTEXT
# subset the hyperparameters for the fasttext models
hyperparameters_ft = HYPERPARAMETERS[HYPERPARAMETERS['emb_type'] == 'ft']

# perform regression analysis with the optimal hyperparameter settings
regression_analysis(merged_ft_df, hyperparameters_ft, FEATURE, TARGET, ITEM, GROUP, ft=True)


# BERT
# subset the hyperparameters for the bert layers
hyperparameters_bert_unprocessed = HYPERPARAMETERS[HYPERPARAMETERS['emb_type'] == 'bert']

# create a dictionary with the name of each association and its corresponding best performing layer
bert_layers = {'feminine' : 1, 'good' : 10, 'smart' : 2, 'trustworthy' : 0}

# initialize an empty dataframe to store hyperparameter information in
hyperparameters_bert = pd.DataFrame(columns = hyperparameters_bert_unprocessed.columns)

# for each association, take the corresponding best-performing hyperparameter settings and save
# it to the dataframe initialized above
for key in bert_layers.keys():
    hyperparameters_bert = pd.concat([hyperparameters_bert, hyperparameters_bert_unprocessed[(hyperparameters_bert_unprocessed['association'] == key) & 
                                                                                             (hyperparameters_bert_unprocessed['emb_model'] == bert_layers[key])]])
# perform regression analysis with the optimal hyperparameter settings
regression_analysis(merged_bert_df, hyperparameters_bert, FEATURE, TARGET, ITEM, GROUP, ft=False)