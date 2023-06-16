from src.embeddings.embedding_fetchers import get_fasttext_embeddings, get_bert_embeddings
from src.analyses.regression_analysis import fetch_ft_embeddings, fetch_bert_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import math
import random
import statistics
import itertools
import pandas as pd



def correlation_analyzer(embedding_df, seed_words_left, seed_words_right):
    assert len(seed_words_left) == len(seed_words_right), f'expected equal number of words in both seed word lists, got: left = {len(seed_words_left)}, right = {len(seed_words_right)}'

    delta_dict = {}

    for word in pd.unique(embedding_df['name']).tolist():
        for model in pd.unique(embedding_df['model']).tolist(): 
            for seed_word_left, seed_word_right in zip(seed_words_left, seed_words_right):
                word_embedding = embedding_df['embedding'].loc[(embedding_df['name'] == word) & (embedding_df['model'] == model)].to_numpy()[0].reshape(1, -1)

                delta = cosine_similarity(word_embedding, seed_word_left[model].reshape(1, -1)) \
                        - cosine_similarity(word_embedding, seed_word_right[model].reshape(1, -1))

                if word in delta_dict.keys():
                    delta_dict[word][model] = delta

                else: 
                    delta_dict[word] = {}
                    delta_dict[word][model] = delta
    
    return delta_dict



def correlation_bootstrapper(embedding_df, seed_words_dict, delta_df, survey_poles, association, emb_model = None):
    # the following function does the bootstrapping analysis, and returns both the raw scores (boot_scores) and
    # a processed df with aggregate scores. The function also saves both the boot_scores dictionary and processed_df
    # as a .bin file and .csv file, respectively.

    name_type = embedding_df['word_type'].to_numpy()[0]
    emb_type = embedding_df['embedding_type'].to_numpy()[0]

    if emb_model == None:
        pass
    else:
        embedding_df = embedding_df.loc[embedding_df['model'] == emb_model]


    # acquire the correct seed words for the correlation analysis given the association
    seed_words_left = seed_words_dict[0]
    seed_words_right = seed_words_dict[1]

    # first get a baseline score using all of the seed words for the correlation analysis
    # as well as one with only the term used in the survey and its direct antonym
    unsampled_seed_words_left = [seed_words_left[s] for s in seed_words_left.keys()]
    unsampled_seed_words_right = [seed_words_right[s] for s in seed_words_right.keys()]

    unsampled_delta_dict = correlation_analyzer(embedding_df, unsampled_seed_words_left, unsampled_seed_words_right)
    survey_poles_delta_dict = correlation_analyzer(embedding_df, [seed_words_left[survey_poles[0]]], [seed_words_right[survey_poles[1]]])

    # calculate how much seed words are needed to get a roughly 50% sample
    words_per_bootstrap = math.ceil(0.5*len(seed_words_left))

    # then, take 250 bootstrapped samples of half of the seed words, and perform the correlation analysis on the samples
    for sample in range(250):
        print(name_type, association, embedding_df['model'].to_numpy()[0], sample)
        random.seed(sample)

        # sample the keys
        samples_left = random.sample(seed_words_left.keys(), k=words_per_bootstrap)
        samples_right = random.sample(seed_words_right.keys(), k=words_per_bootstrap)

        # use the keys to retrieve the embeddings
        sample_words_left = [seed_words_left[s] for s in samples_left]
        sample_words_right = [seed_words_right[s] for s in samples_right]

        # feed the data to the correlation analysis function
        sample_delta_dict = correlation_analyzer(embedding_df, sample_words_left, sample_words_right)

        # then save the correlation score (delta score) to the aggregate dictionary
        for word in sample_delta_dict.keys():
            for model in sample_delta_dict[word].keys():
                delta_df = pd.concat([delta_df, pd.DataFrame({'name' : word, 
                                                              'word_type' : name_type, 
                                                              'association' : association, 
                                                              'delta' : sample_delta_dict[word][model][0][0], 
                                                              'model' : model, 
                                                              'delta_all_names' : unsampled_delta_dict[word][model][0][0], 
                                                              'delta_survey_poles' : survey_poles_delta_dict[word][model][0][0], 
                                                              'wordset_left' : [samples_left], 
                                                              'wordset_right' : [samples_right]})])

    delta_df = delta_df.reset_index(drop=True)

    # save the df for later use
    delta_df.to_csv('./processed_data/analyses/correlation_analysis/{}_{}_{}{}_correlation_bootstrap_scores.csv'.format(name_type, 
                                                                                                                        association, 
                                                                                                                        emb_type,
                                                                                                                        model),
                    index = False)

    return delta_df



def generate_seed_word_embeddings(emb_type : str):
    feminine_words = ['vrouwelijk', 'vrouwelijke', 'vrouwelijkheid', 'feminien', 'vrouw', 'vrouwtje', 'meisje', 'meid', 'oma', 'dochter']
    masculine_words = ['mannelijk', 'mannelijke', 'mannelijkheid', 'masculien', 'man', 'mannetje', 'jongetje', 'jongen', 'opa', 'zoon']

    good_words = ['goed', 'goede', 'beter', 'betere', 'best', 'slechtste', 'positief', 'positieve', 'correct', 'correcte', 'juist', 
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



def bootstrap_scores_csv_maker(scores_dict, name_type : str, emb_type : str):
    # aggregate_delta_dict[association][word][model] = 250 samples

    column_list = ['association', 'word', 'model', 'delta_all_names', 'delta_survey_poles', 'delta_bootstrapped', 'sd_bootstrapped']
    df_list = []

    for association in scores_dict.keys():
        for word in scores_dict[association].keys():
            for model in scores_dict[association][word].keys():
                all_seed_words_score = scores_dict[association][word][model][0]
                survey_poles_score = scores_dict[association][word][model][1]

                avg_bootstrap_score = statistics.mean(scores_dict[association][word][model][2:])
                sd_bootstrap_score = statistics.stdev(scores_dict[association][word][model][2:])

                df_list.append([association, word, model, all_seed_words_score, survey_poles_score, avg_bootstrap_score, sd_bootstrap_score])

    deltas_df = pd.DataFrame(df_list, columns=column_list)
    deltas_df.to_csv('./results/correlation_analysis/' + name_type + '_' + emb_type + '_correlation_delta_scores.csv', index=False)

    return deltas_df



def fetch_prepared_embedding_lists(emb_type, emb_model=None):
    assert emb_type == 'ft' or emb_type == 'bert', "emb_type should be equal to 'ft' for fastText embeddings, or 'bert' for BERT embeddings."
    
    fem_words_embs, mas_words_embs, good_words_embs, bad_words_embs, smart_words_embs, \
    dumb_words_embs, trust_words_embs, untrust_words_embs = fetch_seed_embeddings(emb_type = emb_type, emb_model=emb_model)

    seed_word_list = {'feminine': [fem_words_embs, mas_words_embs], 'good': [good_words_embs, bad_words_embs], 
                         'smart': [smart_words_embs, dumb_words_embs], 'trustworthy': [trust_words_embs, untrust_words_embs]}

    return seed_word_list