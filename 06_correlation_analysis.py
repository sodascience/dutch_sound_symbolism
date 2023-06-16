from src.analyses.correlation_analysis import generate_seed_word_embeddings, correlation_bootstrapper, fetch_prepared_embedding_lists
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

survey_poles = {'feminine': ['vrouwelijk', 'mannelijk'], 'good': ['goed', 'slecht'], 'smart': ['slim', 'dom'], 'trustworthy': ['betrouwbaar', 'onbetrouwbaar']}

# FastText
seed_word_list_ft = fetch_prepared_embedding_lists(emb_type = 'ft', emb_model=None)

for word_type in pd.unique(survey_ft_df['word_type']).tolist():
    for association in seed_word_list_ft.keys():
        for model in pd.unique(survey_ft_df['model']).tolist():
            emb_df_subset = survey_ft_df.loc[(survey_ft_df['word_type'] == word_type) & (survey_ft_df['model'] == model)].reset_index()
            delta_df = pd.DataFrame(columns=['name', 'word_type', 'association', 'delta', 'model', 'delta_all_names', 'delta_survey_poles', 'wordset_left', 'wordset_right'])
            seed_words_subset = seed_word_list_ft[association]
            survey_poles_subset = survey_poles[association]

            delta_df = correlation_bootstrapper(emb_df_subset, seed_words_subset, delta_df, survey_poles_subset, association, emb_model=None)


# BERT
seed_word_list_bert = fetch_prepared_embedding_lists(emb_type = 'bert', emb_model=None)

for word_type in pd.unique(survey_bert_df['word_type']).tolist():
    for association in seed_word_list_bert.keys():
        for model in pd.unique(survey_bert_df['model']).tolist():
            emb_df_subset = survey_bert_df.loc[(survey_bert_df['word_type'] == word_type) & (survey_bert_df['model'] == model)].reset_index()
            delta_df = pd.DataFrame(columns=['name', 'word_type', 'association', 'delta', 'model', 'delta_all_names', 'delta_survey_poles', 'wordset_left', 'wordset_right'])
            seed_words_subset = seed_word_list_bert[association]
            survey_poles_subset = survey_poles[association]
            
            delta_df = correlation_bootstrapper(emb_df_subset, seed_words_subset, delta_df, survey_poles_subset, association, emb_model=None)

