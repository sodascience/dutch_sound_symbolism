from src.embeddings.psycho_embeddings.psycho_embeddings import ContextualizedEmbedder
import pandas as pd
import fasttext

def get_bert_embeddings(words):
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
