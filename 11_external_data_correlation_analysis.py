from src.embeddings.embedding_fetchers import get_bert_embeddings, get_fasttext_embeddings, df_maker
from src.analyses.correlation_analysis import perform_cosine_analyses_and_save_dfs, fetch_prepared_embedding_lists
import pyreadr
import pickle
import pandas as pd

path_df = './processed_data/analyses/dataframes/'

# create_external_data_dfs()

with open(path_df + 'brysbaert_2014_concreteness_ft_df.pkl', 'rb') as f:
    brysbaert_ft_df = pickle.load(f)

with open(path_df + 'brysbaert_2014_concreteness_bert_df.pkl', 'rb') as f:
    brysbaert_bert_df = pickle.load(f)

with open(path_df + 'vankrunkelsven_2022_gender_ft_df.pkl', 'rb') as f:
    vankrunkelsven_ft_df = pickle.load(f)

with open(path_df + 'vankrunkelsven_2022_gender_bert_df.pkl', 'rb') as f:
    vankrunkelsven_bert_df = pickle.load(f)

# generate_concreteness_seed_embeddings()

## ANALYSES
# Concreteness
con_seed_embs_ft = load_prepared_concreteness_seed_embeddings_dict('ft')
con_seed_embs_bert = load_prepared_concreteness_seed_embeddings_dict('bert')

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
fem_seed_embs_ft = fetch_prepared_embedding_lists(emb_type = 'ft', emb_model=None)['feminine']
fem_seed_embs_ft = {'gender' : fem_seed_embs_ft}
fem_seed_embs_bert = fetch_prepared_embedding_lists(emb_type = 'bert', emb_model=None)['feminine']
fem_seed_embs_bert = {'gender' : fem_seed_embs_bert}

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



# FUNCTIONS
def generate_concreteness_seed_embeddings():
    concrete_seed_words = ['concreet', 'concrete', 'concreter', 'concreetheid', 'tastbaar', 'tastbare', 'tastbaarheid', 'duidelijk', 'duidelijke', 
                       'duidelijkheid', 'expliciet', 'expliciete', 'explicietheid', 'materieel', 'materiële', 'grijpbaar', 'grijpbare', 'grijpbaarheid']
    abstract_seed_words = ['abstract', 'abstracte', 'abstracter', 'abstractheid', 'ontastbaar', 'ontastbare', 'ontastbaarheid', 'onduidelijk', 'onduidelijke', 
                        'onduidelijkheid', 'impliciet', 'impliciete', 'implicietheid', 'immaterieel', 'immateriële', 'ongrijpbaar', 'ongrijpbare', 'ongrijpbaarheid']
    con_embs_ft = get_fasttext_embeddings(concrete_seed_words)
    con_embs_bert = get_bert_embeddings(concrete_seed_words)
    abs_embs_ft = get_fasttext_embeddings(abstract_seed_words)
    abs_embs_bert = get_bert_embeddings(abstract_seed_words)
    embs_path = './processed_data/embeddings/'
    with open(embs_path + 'concrete-words_ft_embs.bin', 'wb') as f:
        pickle.dump(con_embs_ft, f)
    with open(embs_path + 'concrete-words_bert_embs.bin', 'wb') as f:
        pickle.dump(con_embs_bert, f)
    with open(embs_path + 'abstract-words_ft_embs.bin', 'wb') as f:
        pickle.dump(abs_embs_ft, f)
    with open(embs_path + 'abstract-words_bert_embs.bin', 'wb') as f:
        pickle.dump(abs_embs_bert, f)


def load_prepared_concreteness_seed_embeddings_dict(model):
    embs_path = './processed_data/embeddings/'
    if model == 'ft':
        with open(embs_path + 'concrete-words_ft_embs.bin', 'rb') as f:
            con_embs_ft = pickle.load(f)
        with open(embs_path + 'abstract-words_ft_embs.bin', 'rb') as f:
            abs_embs_ft = pickle.load(f)
        return {'concreteness' : [con_embs_ft, abs_embs_ft]}
    elif model == 'bert':
        with open(embs_path + 'concrete-words_bert_embs.bin', 'rb') as f:
            con_embs_bert = pickle.load(f)
        with open(embs_path + 'abstract-words_bert_embs.bin', 'rb') as f:
            abs_embs_bert = pickle.load(f)
        return {'concreteness' : [con_embs_bert, abs_embs_bert]}
    else:
        raise Exception("emb_type should be equal to 'ft' for fastText embeddings, or 'bert' for BERT embeddings.")

def create_external_data_dfs():
    brysbaert_df = pyreadr.read_r('./processed_data/brysbaert_2014.rds')[None]
    speed_df = pyreadr.read_r('./processed_data/speed_2021.rds')[None]
    vankrunkelsven_df = pyreadr.read_r('./processed_data/vankrunkelsven_2022.rds')[None]


    brysbaert_ft = get_fasttext_embeddings(brysbaert_df['word'], models = ['0', '2-5'])
    brysbaert_ft_df = df_maker({'real_word' : brysbaert_ft}, 'brysbaert_2014_concreteness', 'ft')

    brysbaert_bert = get_bert_embeddings(brysbaert_df['word'])
    brysbaert_bert_df = df_maker({'real_word' : brysbaert_bert}, 'brysbaert_2014_concreteness', 'bert')


    # I just realized that for the senses, there's no such thing as the 'antonym' or 'opposite' of a sense. So I can't use this data for the correlation analysis
    speed_ft = get_fasttext_embeddings(speed_df['word'], models = ['0', '2-5'])
    speed_ft_df = df_maker({'real_word' : speed_ft}, 'speed_2021_senses', 'ft')

    speed_bert = get_bert_embeddings(speed_df['word'])
    speed_bert_df = df_maker({'real_word' : speed_bert}, 'speed_2021_senses', 'bert')


    vankrunkelsven_ft = get_fasttext_embeddings(vankrunkelsven_df['word'], models = ['0', '2-5'])
    vankrunkelsven_ft_df = df_maker({'real_word' : vankrunkelsven_ft}, 'vankrunkelsven_2022_gender', 'ft')

    vankrunkelsven_bert = get_bert_embeddings(vankrunkelsven_df['word'])
    vankrunkelsven_bert_df = df_maker({'real_word' : vankrunkelsven_bert}, 'vankrunkelsven_2022_gender', 'bert')


