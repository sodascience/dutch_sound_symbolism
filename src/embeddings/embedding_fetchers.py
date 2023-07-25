from src.embeddings.psycho_embeddings.psycho_embeddings import ContextualizedEmbedder
import pandas as pd
import fasttext

def get_bert_embeddings(words):
    '''
    :param words:   list, words/names to get embeddings for

    :return:        dict, dict of dicts containing the embeddings with structure [word][layer][embedding]

    When called, this function opens RobBERT v2 and gets the 'contextless' embeddings (i.e., only a space 
    before and after the target word) for the words in the input list. Then, a dict of dicts is returned.
    '''

    if type(words) != list:
        words = words.tolist()
        
    embedding_texts = [' {0} '.format(word) for word in words]

    model = ContextualizedEmbedder('pdelobelle/robbert-v2-dutch-base', max_length=514)

    embeddings = model.embed(words=words,
                             target_texts=embedding_texts,
                             layers_id=range(12),
                             batch_size=1,
                             averaging=True,
                             return_static=True)

    embs_dict = {}

    for name_index, name in enumerate(words):
        for layer_number in range(12):
            if name in embs_dict.keys():
                embs_dict[name][layer_number] = embeddings[layer_number][name_index]
            else:
                embs_dict[name] = {}
                embs_dict[name][layer_number] = embeddings[layer_number][name_index]

    return embs_dict

def get_fasttext_embeddings(words, models = False):
    '''
    :param words:   list, words/names to get embeddings for
    :paramm models: list or bool, either a list of fasttext models to generate embeddings for. If False, 
                    then all trained models are considered.

    :return:        dict, dict of dicts containing the embeddings with structure [word][model][embedding]

    When called, this function opens the fasttext models to get the embeddings for the words in the input 
    list. Then, a dict of dicts is returned.
    '''

    if models == False:
        models_list = ['0', '2', '2-3', '2-5']
    else:
        models_list = models

    embs_dict = {}

    for model in models_list:
        ft = fasttext.load_model('./processed_data/dsm/corpus_d300_w5_m' + model[0] + '_M' + model[-1] + '.bin')
        for word in words:
            if word in embs_dict.keys():
                embs_dict[word][model] = ft[word]
            else:
                embs_dict[word] = {}
                embs_dict[word][model] = ft[word]

    return embs_dict

def df_maker(embs_dict, data_name, embedding_type):
    '''
    :param embs_dict:       dict, the dict of dicts with structure [word_type][word][model/layer][embedding]
    :param data_name:       str, name of the data to be used when saving the dataframe as a pickle
    :param embedding_type:  str, either 'ft' or 'bert' to indicate whether the df contains fasttext or bert embeddings

    :return:                Pandas dataframe, has the columns name, word_type, embedding_type, model, and embedding. 
                            The dict of dicts are turned into dataframes for easier use in subsequent analyses. The
                            function also automatically saves the dataframe as a .pickle file.
    '''

    df = pd.DataFrame()

    for word_type in embs_dict.keys():
        for name in embs_dict[word_type].keys():
            for model in embs_dict[word_type][name].keys():
                embedding = embs_dict[word_type][name][model]
                df = pd.concat([df, pd.DataFrame({'name':[name], 
                                                  'word_type':[word_type], 
                                                  'embedding_type':[embedding_type], 
                                                  'model':[model], 'embedding':[embedding]})], 
                                axis=0, 
                                ignore_index=True)

    df.to_pickle('./processed_data/analyses/dataframes/' + data_name + '_' + embedding_type + '_df.pkl')

    return df
