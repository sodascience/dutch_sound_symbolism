from src.analyses.regression_analysis import fasttext_analysis, bert_analysis, grids_searcher, open_processed_wordscores_rds
import pickle
import pandas as pd

## LOAD DATA
with open('./processed_data/analyses/dataframes/survey_data_ft_df.pkl', 'rb') as f:
    survey_ft_df = pickle.load(f)   #['name', 'word_type', 'embedding_type', 'model', 'embedding']

with open('./processed_data/analyses/dataframes/survey_data_bert_df.pkl', 'rb') as f:
    survey_bert_df = pickle.load(f)

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
# FASTTEXT
merged_ft_df = pd.merge(survey_ft_df[survey_ft_df['model'].isin(FT_MODELS)], 
                        word_scores, 
                        on=['name', 'word_type'], 
                        how='inner').reindex(
                        columns = ['name', 'word_type', 'association', 'mean_rating', 'embedding_type', 'model', 'embedding'])

grids_searcher(df = merged_ft_df,
               associations = ASSOCIATIONS, 
               x_col = FEATURE, 
               y_col = TARGET, 
               group_col = GROUP, 
               emb_model = FT_MODELS)

# BERT
merged_bert_df = pd.merge(survey_bert_df[survey_bert_df['model'].isin(BERT_LAYERS)], 
                          word_scores, 
                          on=['name', 'word_type'], 
                          how='inner').reindex(
                          columns = ['name', 'word_type', 'association', 'mean_rating', 'embedding_type', 'model', 'embedding'])

grids_searcher(df = merged_bert_df,
               associations = ASSOCIATIONS, 
               x_col = FEATURE, 
               y_col = TARGET, 
               group_col = GROUP, 
               emb_model = BERT_LAYERS)

## REGRESSION ANALYSIS
# FASTTEXT

# hyperparameter_df = pd.DataFrame(columns = ['association', 'word_type', 'embedding_type', 'emb_model', 'units', 'dropout', 'act', 'n_layers', 'lr'])

# regression_analysis(df, hyperparameter_df, associations, x_col, y_col, items_col, group_col, emb_models, units, dropout, act, n_layers, lr)


# BERT