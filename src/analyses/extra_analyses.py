from src.embeddings.embedding_fetchers import get_bert_embeddings, get_fasttext_embeddings, df_maker
import pickle
import pyreadr

def generate_concreteness_seed_embeddings():
    '''
    :return:            none

    This function contains lists of seed words for the left and right side of the semantic poles for our extra analyses. 
    Calling this function automatically generates embeddings for these words and saves them as a .pickle file.
    '''

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
    '''
    :param model:   str or int, must be equal to 'ft' for fasttext embeddings or 'bert' for bert embeddings
    
    :return:        dicts, returns the seed words for the left and right pole of the concreteness association

    Calling this function opens the .pickle files that contain the word embeddings for the seed words and returns 
    them as a dict of dicts.
    '''

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
    '''
    :return:    none

    This function opens the .rds files of the processed data for the brysbaert, speed, and vankrunkelsven datasets.
    Then, fasttext and bert embeddings are generated, of which dataframes are created and saved as .csv files.    
    '''

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