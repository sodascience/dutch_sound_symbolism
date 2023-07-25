from src.embeddings.embedding_fetchers import get_fasttext_embeddings, get_bert_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import math
import random
import statistics
import itertools
import pandas as pd
import numpy as np



def perform_cosine_analyses_and_save_dfs(input_df, seed_words, emb_type, word_types, associations, models, bootstrap=False):
    '''
    :param input_df:        Pandas DataFrame, contains the name, word type, embedding type (ft or bert), model/layer and the 
                            word embedding for the words to be analyzed
    :param seed_words:      dict, contains a dict of dicts with structure [name][model/layer][embedding] for the seed words 
                            used in the analysis 
    :param emb_type:        str, whether the embeddings are from fastText (ft) or BERT (bert)
    :param word_types:      list, all unique word types of the names and words in the input_df
    :param associations:    list, all distinct associations on which the input_df words will be tested
    :param models:          list, all fasttext models / bert layers that have been used to generate embeddings, and which will 
                            be compared/tested in the analysis
    :param bootstrap:       bool, if true, random bootstrap samples will be drawn to measure errors     

    :return:                none, .csv files are saved automatically per unique association
    
    Loops over the association, word_types, and models to feed a subset of the data 
    to the cosine_var_prepper() for the cosine analysis, .csv files are automatically saved.
    '''
    
    if (emb_type == 'bert' and len(models) == 12):
        # if all bert layers are used, it's less messy to just add '-all-layers' instead of '1_&_2_&_3_' etc
        model_types = '-all-layers'
    else:
        model_types = '_&_'.join(models)

    for association in associations:
        # for each iteration, get a subset of the seed_words to only get those seed words/poles related to the association at hand
        seed_words_subset = seed_words[association]
        
        # get a list of only the actual seed words, not with any embeddings attached
        seed_word_list = list(seed_words_subset[0].keys()) + list(seed_words_subset[1].keys())

        # create a list of the column names for our dataframe
        if bootstrap == True:
            column_names = ['name', 'word_type', 'association', 'model', 'delta', 'delta_all_names'] + seed_word_list

        else:
            column_names = ['name', 'word_type', 'association', 'model', 'delta_all_names']

        # initialize an empty data frame for all sub-conditions of the current association
        deltas_df = pd.DataFrame(columns = column_names)

        for word_type in word_types:
            for model in models:
                # get a subset of the input_df dataframe, to only get the slice of the df related to the word type and model at hand
                emb_df_subset = input_df.loc[(input_df['word_type'] == word_type) & (input_df['model'] == model)].reset_index()

                # initialize an empty data frame for the subsample
                delta_df = pd.DataFrame(columns=column_names)
                
                # feed data to the cosine prep function
                delta_df = cosine_var_preparer(emb_df_subset, seed_words_subset, delta_df, association, emb_model=model, bootstrap=bootstrap)

                # add the results to the dataframe
                deltas_df = pd.concat([deltas_df, delta_df])

        # save the dataframe for this specific association as a .csv file
        deltas_df.to_csv('./processed_data/analyses/correlation_analysis/{}_{}{}_bootstrap={}_cosine_scores.csv'.format(association, emb_type, model_types, bootstrap), index=False)



    
def cosine_var_preparer(embedding_df, seed_words_dict, delta_df, association, emb_model = None, bootstrap = False):
    '''
    :param embedding_df:    Pandas DataFrame, contains the subset (based on the current association, word_type, 
                            and model) created in the perform_cosine_analysis_and_save_dfs() function. Has the name, 
                            word type, embedding type (ft or bert), model/layer and the word embedding for the words 
                            to be analyzed
    :param seed_words_dic:  dict, contains a dict of dicts with structure [left(0)/right(1) side of semantic pole]
                            [name][model/layer][embedding] for the seed words used in the analysis 
    :param delta_df:        Pandas DataFrame, empty DF to save the cosine output in
    :param association:     str, associations on which the embedding_df words will be tested
    :param emb_model:       str, fasttext model / bert layer that have been used to generate the word embeddings
    :param bootstrap:       bool, if true, random bootstrap samples will be drawn to measure errors     

    :return:                Pandas DataFrame, contains the word, name type, association, model/layer, and cosine delta 
                            score per word in the embedding_df
    
    This function prepares the variables and dfs to be fed to the looper function, and if bootstrap==True,
    returns both the raw scores (boot_scores) and a processed df with aggregate scores.
    '''

    name_type = embedding_df['word_type'].to_numpy()[0]

    if emb_model == None:
        pass
    else:
        embedding_df = embedding_df.loc[embedding_df['model'] == emb_model]


    # acquire the correct seed words for the cosine analysis given the association
    seed_words_left = seed_words_dict[0]
    seed_words_right = seed_words_dict[1]

    n_seed_words = len(seed_words_left.keys()) + len(seed_words_right.keys())

    # first get a baseline score using all of the seed words for the cosine analysis with cosine_looper()
    unsampled_seed_words_left = [seed_words_left[s] for s in seed_words_left.keys()]
    unsampled_seed_words_right = [seed_words_right[s] for s in seed_words_right.keys()]

    unsampled_delta_dict = cosine_looper(embedding_df, unsampled_seed_words_left, unsampled_seed_words_right)

    if bootstrap == True:
        # if you want to perform bootstrap analysis, all relevant info is fed to the cosine_bootstrapper() function
        delta_df = cosine_bootstrapper(embedding_df,
                                       delta_df,
                                       unsampled_delta_dict, 
                                       seed_words_left,
                                       seed_words_right,
                                       n_seed_words,
                                       name_type,
                                       association)

        return delta_df
    
    else:
        # if no bootstrap analysis is performed, iteratively build the output dataframe for each unique word and model
        for word in unsampled_delta_dict.keys():
            for model in unsampled_delta_dict[word].keys():
                # prepare the row for the delta_df
                row = [word, name_type, association, model, unsampled_delta_dict[word][model]]
                # prepare the column names
                col_names = list(delta_df.columns)

                # create a temporary dataframe for the current row
                temp_df = pd.DataFrame(data = [row], columns = col_names)
                
                # concatenate the current row to the full dataframe
                delta_df = pd.concat([delta_df, temp_df])
        
        return delta_df



def cosine_looper(embedding_df, seed_words_left, seed_words_right):
    '''
    :param embedding_df:        Pandas DataFrame, contains the subset (based on the current association, word_type, 
                                and model) created in the perform_cosine_analysis_and_save_dfs() function. Has the name, 
                                word type, embedding type (ft or bert), model/layer and the word embedding for the words 
                                to be analyzed
    :param seed_words_left:     dict, contains a dict of dicts with structure [name][model/layer][embedding] for the seed 
                                words on the left side of the semantic pole used in the analysis 
    :param seed_words_right:    dict, contains a dict of dicts with structure [name][model/layer][embedding] for the seed 
                                words on the right side of the semantic pole used in the analysis 
    
    :return:                    dict, contains a dict of dicts with delta scores for all target words using all seed words,
                                with the structure [name][model/layer][cosine delta score]
    
    This function loops over the unique words and models in the embedding_df, and calls the cosine_calculator() function
    with the word_embedding and seed words to calculate the target word's semantic pole delta score. Then saves this 
    score in the delta_dict with the structure [word][ft model/bert layer][delta score]
    '''

    # first assert that the length of both seed word lists are the same. If not, raise error.
    assert len(seed_words_left) == len(seed_words_right), f'expected equal number of words in both seed word lists, got: left = {len(seed_words_left)}, right = {len(seed_words_right)}'

    # initialize empty dictionary to save the delta scores to
    delta_dict = {}

    for word in pd.unique(embedding_df['name']).tolist():
        for model in pd.unique(embedding_df['model']).tolist(): 
            # for each unique word and model, fetch the embedding from the embedding_df
            word_embedding = embedding_df['embedding'].loc[(embedding_df['name'] == word) & (embedding_df['model'] == model)].to_numpy()[0].reshape(1, -1)
            
            # get the delta score by calling the cosine calculator function with the embedding and seed words
            delta = cosine_calculator(word_embedding, seed_words_left, seed_words_right, model)

            if word in delta_dict.keys():
                # if the word already exists as a key in the delta_dict, add a new entry to the dict of dicts
                delta_dict[word][model] = delta

            else: 
                # else, add the key and assign it an empty dictionary, and then add the new entry
                delta_dict[word] = {}
                delta_dict[word][model] = delta
    
    return delta_dict




def cosine_calculator(word_embedding, seed_words_left, seed_words_right, model):
    '''
    :param word_embedding:      Numpy array, contains the word embedding of the target word
    :param seed_words_left:     dict, contains a dict of dicts with structure [name][model/layer][embedding] for the seed 
                                words on the left side of the semantic pole used in the analysis 
    :param seed_words_right:    dict, contains a dict of dicts with structure [name][model/layer][embedding] for the seed 
                                words on the right side of the semantic pole used in the analysis 
    
    :return:                    float, delta score
    
    This function calculates the cosine similarity between the target word and all seed words (left and right). Then, the 
    cosine similarities are averaged for the left pole and the right pole, after which the delta score (left - right) is 
    calculated and returned
    '''

    # prepare two lists to append the cosine similarity scores with all seed words to
    cosine_similarities_left = []
    cosine_similarities_right = []

    for seed_word_left, seed_word_right in zip(seed_words_left, seed_words_right):
        # for each seed word, calculate the cosine similarity between the target word/name and the seed word, append to the list
        cosine_similarities_left.append(cosine_similarity(word_embedding, seed_word_left[model].reshape(1, -1)))
        cosine_similarities_right.append(cosine_similarity(word_embedding, seed_word_right[model].reshape(1, -1)))

    # calculate the difference (delta) between the average cosine similarity of the target word/name with the 'left' seed words and the 'right' seed words
    delta = np.mean(cosine_similarities_left) - np.mean(cosine_similarities_right)

    return delta




def cosine_bootstrapper(embedding_df, delta_df, unsampled_delta_dict, seed_words_left, seed_words_right, n_seed_words, name_type, association):
    '''
    :param embedding_df:            Pandas DataFrame, contains the subset (based on the current association, word_type, 
                                    and model) created in the perform_cosine_analysis_and_save_dfs() function. Has the name, 
                                    word type, embedding type (ft or bert), model/layer and the word embedding for the words 
                                    to be analyzed
    :param delta_df:                Pandas DataFrame, empty DF to save the cosine output in
    :param unsampled_delta_dict:    dict, contains a dict of dicts with delta scores for all target words using all seed words,
                                    with the structure [name][model/layer][cosine delta score] 
    :param seed_words_left:         dict, contains a dict of dicts with structure [name][model/layer][embedding] for the seed 
                                    words on the left side of the semantic pole used in the analysis 
    :param seed_words_right:        dict, contains a dict of dicts with structure [name][model/layer][embedding] for the seed 
                                    words on the right side of the semantic pole used in the analysis 
    :param n_seed_words:            int, number of total seed words in analysis (left and right)
    :param name_type:               str, name type of the target words/names 
    :param association:             str, associations on which the embedding_df words will be tested
    
    :return:                        Pandas DataFrame, contains the word, name type, association, model/layer, cosine delta 
                                    score, and a list of columns with either a 0 or a 1, depending on whether they were used 
                                    in a specific bootstrap sample, per word in the embedding_df and one of the 250 bootstrap
                                    samples
    
    This function prepares the variables and dfs to be fed to the looper function, and if bootstrap==True,
    returns both the raw scores (boot_scores) and a processed df with aggregate scores.
    '''

   
    # calculate how much seed words are needed to get a roughly 50% sample
    words_per_bootstrap = math.ceil(0.5*len(seed_words_left))

    # then, take 250 bootstrapped samples of half of the seed words, and perform the cosine analysis on the samples
    for sample in range(250):
        print(name_type, association, embedding_df['model'].to_numpy()[0], sample)
        random.seed(sample)

        # sample the keys
        samples_left = random.sample(seed_words_left.keys(), k=words_per_bootstrap)
        samples_right = random.sample(seed_words_right.keys(), k=words_per_bootstrap)

        # use the keys to retrieve the embeddings
        sample_words_left = [seed_words_left[s] for s in samples_left]
        sample_words_right = [seed_words_right[s] for s in samples_right]

        # feed the data to the cosine analysis function
        sample_delta_dict = cosine_looper(embedding_df, sample_words_left, sample_words_right)

        # then save the cosine score (delta score) to the aggregate dictionary
        for word in sample_delta_dict.keys():
            for model in sample_delta_dict[word].keys():
                # create a list that will be entered as a row in the dataframe
                row = [word, name_type, association, model, 
                       sample_delta_dict[word][model], 
                       unsampled_delta_dict[word][model]] + [0] * n_seed_words
                
                # get the column names
                col_names = list(delta_df.columns) 
                # get a list of the sampled seed words
                sample_words = samples_left + samples_right
                
                # create a temporary df of the row to be added to the full df
                temp_df = pd.DataFrame(data = [row], columns = col_names)
                # for all of the sampled seed words, assign the value 1 to their respective columns
                temp_df[sample_words] = 1

                # concatenate the row with the full df
                delta_df = pd.concat([delta_df, temp_df])
    
    return delta_df.reset_index(drop=True)




def generate_seed_word_embeddings(emb_type : str):
    '''
    :param emb_type:    str, must be equal to 'ft' or 'bert', to generate fasttext or bert embeddings for the seed words

    :return:            none

    This function contains lists of seed words for the left and right side of the semantic poles for our analyses. 
    Calling this function automatically generates embeddings for these words and saves them as a .pickle file.
    '''
    
    feminine_words = ['vrouwelijk', 'vrouwelijke', 'vrouwelijkheid', 'feminien', 'vrouw', 'vrouwtje', 'meisje', 'meid', 'oma', 'dochter']
    masculine_words = ['mannelijk', 'mannelijke', 'mannelijkheid', 'masculien', 'man', 'mannetje', 'jongetje', 'jongen', 'opa', 'zoon']

    good_words = ['goed', 'goede', 'beter', 'betere', 'best', 'beste', 'positief', 'positieve', 'correct', 'correcte', 'juist', 
                'juiste', 'goedaardig', 'goedaardige', 'aardig', 'aardige', 'lief', 'lieve', 'moreel', 'morele']
    bad_words = ['slecht', 'slechte', 'slechter', 'slechtere', 'slechtst', 'slechtste', 'negatief', 'negatieve', 'incorrect', 'incorrecte', 'onjuist', 
                'onjuiste', 'kwaadaardig', 'kwaadaardige', 'onaardig', 'onaardige', 'gemeen', 'gemene', 'immoreel', 'immorele']

    smart_words = ['slim', 'slimme', 'intelligent', 'intelligente', 'verstandig', 'verstandige', 'opmerkzaam', 'opmerkzame', 'begaafd', 'begaafde', 
                'geleerd', 'geleerde', 'gevat', 'gevatte', 'snugger', 'snuggere', 'scherpzinnig', 'scherpzinnige', 'pienter', 'pientere']
    dumb_words = ['dom', 'domme', 'onintelligent', 'onintelligente', 'onverstandig', 'onverstandinge', 'onopmerkzaam', 'onopmerkzame', 'ongebaafd', 'onbegaafde', 
                'ongeleerd', 'ongeleerde', 'onnozel', 'onnozele', 'dwaas', 'dwaze', 'zot', 'zotte', 'onbenullig', 'onbenullige']

    trustworthy_words = ['betrouwbaar', 'betrouwbare', 'eerlijk', 'eerlijke', 'bonafide', 'beproefd', 'beproefde', 'degelijk', 'degelijke', 
                        'gegrond', 'gegronde', 'integer', 'integere', 'trouw', 'trouwe', 'loyaal', 'loyale', 'veilig', 'veilige', 'solide']
    untrustworthy_words = ['onbetrouwbaar', 'onbetrouwbare', 'oneerlijk', 'oneerlijke', 'malafide', 'onbeproefd', 'onbeproefde', 'ondegelijk', 'ondegelijke',
                        'ongegrond', 'ongegronde', 'corrupt', 'corrupte', 'ontrouw', 'ontrouwe', 'achterbaks', 'achterbakse', 'onveilig', 'onveilige', 'onsolide']
    
    if emb_type  == 'ft':
        fem_words_embs = get_fasttext_embeddings(feminine_words)
        mas_words_embs = get_fasttext_embeddings(masculine_words)

        good_words_embs = get_fasttext_embeddings(good_words)
        bad_words_embs = get_fasttext_embeddings(bad_words)

        smart_words_embs = get_fasttext_embeddings(smart_words)
        dumb_words_embs = get_fasttext_embeddings(dumb_words)

        trust_words_embs = get_fasttext_embeddings(trustworthy_words)
        untrust_words_embs = get_fasttext_embeddings(untrustworthy_words)

    elif emb_type == 'bert':
        fem_words_embs = get_bert_embeddings(feminine_words)
        mas_words_embs = get_bert_embeddings(masculine_words)

        good_words_embs = get_bert_embeddings(good_words)
        bad_words_embs = get_bert_embeddings(bad_words)

        smart_words_embs = get_bert_embeddings(smart_words)
        dumb_words_embs = get_bert_embeddings(dumb_words)

        trust_words_embs = get_bert_embeddings(trustworthy_words)
        untrust_words_embs = get_bert_embeddings(untrustworthy_words)

    else:
        raise Exception("emb_type should be equal to 'ft' for fastText embeddings, or 'bert' for BERT embeddings.")

    with open('./processed_data/embeddings/feminine-words_' + emb_type + '_embs.bin', 'wb') as f:
        pickle.dump(fem_words_embs, f)
    with open('./processed_data/embeddings/masculine-words_' + emb_type + '_embs.bin', 'wb') as f:
        pickle.dump(mas_words_embs, f)
    with open('./processed_data/embeddings/good-words_' + emb_type + '_embs.bin', 'wb') as f:
        pickle.dump(good_words_embs, f)
    with open('./processed_data/embeddings/bad-words_' + emb_type + '_embs.bin', 'wb') as f:
        pickle.dump(bad_words_embs, f)
    with open('./processed_data/embeddings/smart-words_' + emb_type + '_embs.bin', 'wb') as f:
        pickle.dump(smart_words_embs, f)
    with open('./processed_data/embeddings/dumb-words_' + emb_type + '_embs.bin', 'wb') as f:
        pickle.dump(dumb_words_embs, f)
    with open('./processed_data/embeddings/trustworthy-words_' + emb_type + '_embs.bin', 'wb') as f:
        pickle.dump(trust_words_embs, f)
    with open('./processed_data/embeddings/untrustworthy-words_' + emb_type + '_embs.bin', 'wb') as f:
        pickle.dump(untrust_words_embs, f)


def fetch_seed_embeddings(emb_type, emb_model=None):
    '''
    :param emb_type:    str, must be equal to 'ft' or 'bert', to fetch fasttext or bert embeddings for the seed words
    :param emb_model:   str or int, must be equal to one of the ft models or bert layers, to fetch embeddings for 
                        that specific model/layer only
    
    :return:            dicts, returns 8 dictionaries of word embeddings for the two semantic poles of the four 
                        associations that we gathered data for in our survey

    Calling this function opens the .pickle files that contain the word embeddings for the seed words and returns them
    to the fetch_prepared_embedding_lists() to turn it into a nested dictionary to be used in the analyses.
    '''

    with open('./processed_data/embeddings/feminine-words_' + emb_type + '_embs.bin', 'rb') as f:
        fem_words_embs = pickle.load(f)
    with open('./processed_data/embeddings/masculine-words_' + emb_type + '_embs.bin', 'rb') as f:
        mas_words_embs = pickle.load(f)
    with open('./processed_data/embeddings/good-words_' + emb_type + '_embs.bin', 'rb') as f:
        good_words_embs = pickle.load(f)
    with open('./processed_data/embeddings/bad-words_' + emb_type + '_embs.bin', 'rb') as f:
        bad_words_embs = pickle.load(f)
    with open('./processed_data/embeddings/smart-words_' + emb_type + '_embs.bin', 'rb') as f:
        smart_words_embs = pickle.load(f)
    with open('./processed_data/embeddings/dumb-words_' + emb_type + '_embs.bin', 'rb') as f:
        dumb_words_embs = pickle.load(f)
    with open('./processed_data/embeddings/trustworthy-words_' + emb_type + '_embs.bin', 'rb') as f:
        trust_words_embs = pickle.load(f)
    with open('./processed_data/embeddings/untrustworthy-words_' + emb_type + '_embs.bin', 'rb') as f:
        untrust_words_embs = pickle.load(f)

    if emb_model != None:
        for dictionary in [fem_words_embs, mas_words_embs, good_words_embs, bad_words_embs, smart_words_embs, dumb_words_embs, trust_words_embs, untrust_words_embs]:
            for word in dictionary.keys():
                dictionary[word] = dictionary[word][emb_model]
        
        return fem_words_embs, mas_words_embs, good_words_embs, bad_words_embs, smart_words_embs, dumb_words_embs, trust_words_embs, untrust_words_embs

    else:
        return fem_words_embs, mas_words_embs, good_words_embs, bad_words_embs, smart_words_embs, dumb_words_embs, trust_words_embs, untrust_words_embs


def fetch_prepared_embedding_lists(emb_type, emb_model=None):
    '''
    :param emb_type:    str, must be equal to 'ft' or 'bert', to fetch fasttext or bert embeddings for the seed words
    :param emb_model:   str or int, must be equal to one of the ft models or bert layers, to fetch embeddings for 
                        that specific model/layer only
    
    :return:            dicts, returns 8 dictionaries of word embeddings for the two semantic poles of the four 
                        associations that we gathered data for in our survey

    This function calls the fetch_seed_embeddings() function and uses the dicts that are returned to create a nested 
    dictionary to be used in the cosine and correlation analyses.
    '''
    
    assert emb_type == 'ft' or emb_type == 'bert', "emb_type should be equal to 'ft' for fastText embeddings, or 'bert' for BERT embeddings."
    
    fem_words_embs, mas_words_embs, good_words_embs, bad_words_embs, smart_words_embs, \
    dumb_words_embs, trust_words_embs, untrust_words_embs = fetch_seed_embeddings(emb_type = emb_type, emb_model=emb_model)

    seed_word_list = {'feminine': [fem_words_embs, mas_words_embs], 'good': [good_words_embs, bad_words_embs], 
                         'smart': [smart_words_embs, dumb_words_embs], 'trustworthy': [trust_words_embs, untrust_words_embs]}

    return seed_word_list