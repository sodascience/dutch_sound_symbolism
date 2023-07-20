'''
This script is used to generate data for the correlation analysis and grid search
using the open access data gathered by other researchers. The data generated using 
this script can be analyzed using script 12.

Note: due to time constraints, regression analyses using the grid search results 
were not implemented for the open access data
'''

from src.analyses.extra_analyses import create_external_data_dfs, generate_concreteness_seed_embeddings, load_prepared_concreteness_seed_embeddings_dict
from src.analyses.correlation_analysis import perform_cosine_analyses_and_save_dfs, fetch_prepared_embedding_lists
from src.analyses.regression_analysis import grids_searcher
import pyreadr
import pickle
import pandas as pd

path_df = './processed_data/analyses/dataframes/'

# using the external data, create cleaned dataframes that can be used for our analyses
create_external_data_dfs()

# open the dataframe pickle files
with open(path_df + 'brysbaert_2014_concreteness_ft_df.pkl', 'rb') as f:
    brysbaert_ft_df = pickle.load(f)

with open(path_df + 'brysbaert_2014_concreteness_bert_df.pkl', 'rb') as f:
    brysbaert_bert_df = pickle.load(f)

with open(path_df + 'vankrunkelsven_2022_gender_ft_df.pkl', 'rb') as f:
    vankrunkelsven_ft_df = pickle.load(f)

with open(path_df + 'vankrunkelsven_2022_gender_bert_df.pkl', 'rb') as f:
    vankrunkelsven_bert_df = pickle.load(f)

# generate embeddings for the seed words used for the concreteness correlation analysis
generate_concreteness_seed_embeddings()

## CORRELATION ANALYSIS
# Concreteness
# open fasttext and bert seed word embedding dictionaries
con_seed_embs_ft = load_prepared_concreteness_seed_embeddings_dict('ft')
con_seed_embs_bert = load_prepared_concreteness_seed_embeddings_dict('bert')

# perform the cosine analyses and save the dataframes for fasttext and BERT
perform_cosine_analyses_and_save_dfs(input_df = brysbaert_ft_df, 
                                     seed_words = con_seed_embs_ft, 
                                     emb_type = 'ft', 
                                     word_types = pd.unique(brysbaert_ft_df['word_type']).tolist(), 
                                     associations = con_seed_embs_ft.keys(), 
                                     models = ['0', '2-5'],
                                     bootstrap = False)

perform_cosine_analyses_and_save_dfs(input_df = brysbaert_bert_df, 
                                     seed_words = con_seed_embs_bert, 
                                     emb_type = 'bert', 
                                     word_types = pd.unique(brysbaert_bert_df['word_type']).tolist(), 
                                     associations = con_seed_embs_bert.keys(), 
                                     models = list(range(12)),
                                     bootstrap = False)

# Gender
# open fasttext and bert seed word embedding dictionaries created for the (femininity)
# correlation analysis, pick the feminine data and label it as 'gender' in a new dictionary
fem_seed_embs_ft = fetch_prepared_embedding_lists(emb_type = 'ft', emb_model=None)['feminine']
fem_seed_embs_ft = {'gender' : fem_seed_embs_ft}
fem_seed_embs_bert = fetch_prepared_embedding_lists(emb_type = 'bert', emb_model=None)['feminine']
fem_seed_embs_bert = {'gender' : fem_seed_embs_bert}

# perform the cosine analyses and save the dataframes for fasttext and BERT
perform_cosine_analyses_and_save_dfs(input_df = vankrunkelsven_ft_df, 
                                     seed_words = fem_seed_embs_ft, 
                                     emb_type = 'ft', 
                                     word_types = pd.unique(vankrunkelsven_ft_df['word_type']).tolist(), 
                                     associations = fem_seed_embs_ft.keys(), 
                                     models = ['0', '2-5'],
                                     bootstrap = False)

perform_cosine_analyses_and_save_dfs(input_df = vankrunkelsven_bert_df, 
                                     seed_words = fem_seed_embs_bert, 
                                     emb_type = 'bert', 
                                     word_types = pd.unique(vankrunkelsven_bert_df['word_type']).tolist(), 
                                     associations = fem_seed_embs_bert.keys(), 
                                     models = list(range(12)),
                                     bootstrap = False)


## REGRESSION ANALYSIS
# SET IMPORTANT PARAMETERS
M = 2
RND_ITERS = 50
FEATURE = 'embedding'
TARGET = 'mean_rating'
GROUP = 'word_type'
ITEM = 'name'
FT_MODELS = ['0', '2-5']
BERT_LAYERS = list(range(12))

# Concreteness
# load the word norms data and rename columns
brysbaert_scores_df = pyreadr.read_r('./processed_data/brysbaert_2014.rds')[None]
brysbaert_scores_df = brysbaert_scores_df.rename(columns = {'word' : 'name',
                                                            'concreteness_brysbaert' : 'mean_rating'})

# fastText
# merge the word norms data and the word embeddings into a single dataframe
merged_con_ft_df = pd.merge(brysbaert_ft_df[brysbaert_ft_df['model'].isin(FT_MODELS)], 
                            brysbaert_scores_df, 
                            on=['name'], 
                            how='inner').reindex(
                            columns = ['name', 'word_type', 'association', 'mean_rating', 'embedding_type', 'model', 'embedding'])
merged_con_ft_df['association'] = 'concreteness'

# perform the grid search
grids_searcher(df = merged_con_ft_df.dropna(), # Takes soooo long, I can also use most common ft parameters from other grid search best models,
               associations = ['concreteness'], # so, units = 25, dropout = 0.50, activation = sigmoid, n_layers = 1, lr = 0.001
               x_col = FEATURE, # basically same stuff for BERT, but then also have to choose a layer (best performing overall = )
               y_col = TARGET, 
               group_col = GROUP, 
               emb_model = FT_MODELS) 


# BERT
# merge the word norms data and the word embeddings into a single dataframe
merged_con_bert_df = pd.merge(brysbaert_bert_df[brysbaert_bert_df['model'].isin(BERT_LAYERS)], 
                              brysbaert_scores_df, 
                              on=['name'], 
                              how='inner').reindex(
                              columns = ['name', 'word_type', 'association', 'mean_rating', 'embedding_type', 'model', 'embedding'])
merged_con_bert_df['association'] = 'concreteness'

# perform the grid search
grids_searcher(df = merged_con_bert_df.dropna(),
               associations = ['concreteness'], 
               x_col = FEATURE, 
               y_col = TARGET, 
               group_col = GROUP, 
               emb_model = BERT_LAYERS)


# Gender
# load the word norms data and rename columns
vankrunkelsven_scores_df = pyreadr.read_r('./processed_data/vankrunkelsven_2022.rds')[None]
vankrunkelsven_scores_df = vankrunkelsven_scores_df.rename(columns = {'word' : 'name',
                                                                      'gender_vankrunkelsven' : 'mean_rating'})

# fastText
# merge the word norms data and the word embeddings into a single dataframe
merged_fem_ft_df = pd.merge(vankrunkelsven_ft_df[vankrunkelsven_ft_df['model'].isin(FT_MODELS)], 
                            vankrunkelsven_scores_df, 
                            on=['name'], 
                            how='inner').reindex(
                            columns = ['name', 'word_type', 'association', 'mean_rating', 'embedding_type', 'model', 'embedding'])
merged_fem_ft_df['association'] = 'gender'

# perform the grid search
grids_searcher(df = merged_fem_ft_df.dropna(),
               associations = ['gender'], 
               x_col = FEATURE, 
               y_col = TARGET, 
               group_col = GROUP, 
               emb_model = FT_MODELS)

# BERT
# merge the word norms data and the word embeddings into a single dataframe
merged_fem_bert_df = pd.merge(vankrunkelsven_bert_df[vankrunkelsven_bert_df['model'].isin(BERT_LAYERS)], 
                              vankrunkelsven_scores_df, 
                              on=['name'], 
                              how='inner').reindex(
                              columns = ['name', 'word_type', 'association', 'mean_rating', 'embedding_type', 'model', 'embedding'])
merged_fem_bert_df['association'] = 'gender'

# perform the grid search
grids_searcher(df = merged_fem_bert_df.dropna(),
               associations = ['gender'], 
               x_col = FEATURE, 
               y_col = TARGET, 
               group_col = GROUP, 
               emb_model = BERT_LAYERS)

