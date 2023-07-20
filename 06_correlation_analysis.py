from src.analyses.correlation_analysis import generate_seed_word_embeddings, perform_cosine_analyses_and_save_dfs, fetch_prepared_embedding_lists
import pickle
import itertools
import pandas as pd

## LOAD DATA

with open('./processed_data/analyses/dataframes/survey_data_ft_df.pkl', 'rb') as f:
    survey_ft_df = pickle.load(f)   #['name', 'word_type', 'embedding_type', 'model', 'embedding']

with open('./processed_data/analyses/dataframes/survey_data_bert_df.pkl', 'rb') as f:
    survey_bert_df = pickle.load(f)


## CORRELATION ANALYSIS

# for list of seed words for correlation analysis see:
# ./src/analysis/correlation_analysis.py

# generate_seed_word_embeddings('ft')
# generate_seed_word_embeddings('bert')

# FastText
seed_word_list_ft = fetch_prepared_embedding_lists(emb_type = 'ft', emb_model=None)

perform_cosine_analyses_and_save_dfs(input_df = survey_ft_df, 
                                     seed_words = seed_word_list_ft, 
                                     emb_type = 'ft', 
                                     word_types = pd.unique(survey_ft_df['word_type']).tolist(), 
                                     associations = seed_word_list_ft.keys(), 
                                     models = ['0', '2-5'],
                                     bootstrap = False)


# BERT
seed_word_list_bert = fetch_prepared_embedding_lists(emb_type = 'bert', emb_model=None)

perform_cosine_analyses_and_save_dfs(input_df = survey_bert_df, 
                                     seed_words = seed_word_list_bert, 
                                     emb_type = 'bert', 
                                     word_types = pd.unique(survey_bert_df['word_type']).tolist(), 
                                     associations = seed_word_list_bert.keys(), 
                                     models = pd.unique(survey_bert_df['model']).tolist(), 
                                     bootstrap = True)
