'''
This script can be used to train fasttext models using the cleaned corpus file. The four options chosen in this file
include a purely lexical model (i.e., word2vec, no sub-word information is used), a bi-gram model (which only trains 
lexical embeddings and embeddings for bigrams), a bi- and tri-gram model, and a model that trains lexical embeddings 
as well as embeddings for 2-grams, 3-grams, 4-grams, and 5-grams. Models are saved as .bin files in the form of 
'[corpus name]_d[dimensionality]_w[window size]_m[minimum n-gram size]_M[maximum n-gram size].bin'
'''

import fasttext
import os

# set constant variables
corpus = './processed_data/corpus/corpus_final.txt'
dim_best = 300
ws_best = 5
min_n_best = 2
max_n_best = 3
min_n_lexical = 0
max_n_lexical = 0
out_path = './processed_data/dsm/'

# train and save lexical model
model_lexical = fasttext.train_unsupervised(input = corpus,
                                            model = "skipgram",
                                            dim = dim_best,
                                            ws = ws_best,
                                            minn = min_n_lexical,
                                            maxn = max_n_lexical,
                                            lr = 0.01,
                                            epoch = 10,
                                            minCount = 3,
                                            thread = 8,
                                            bucket=4000000)

model_lexical.save_model(os.path.join(
                        out_path, '{}_d{}_w{}_m{}_M{}.bin'.format(
                        os.path.splitext(corpus)[0].split('/')[1], dim_best, ws_best, min_n_lexical, max_n_lexical)))
del model_lexical

# train and save 2-3-gram model
model_subword = fasttext.train_unsupervised(input = corpus,
                                            model = "skipgram",
                                            dim = dim_best,
                                            ws = ws_best,
                                            minn = min_n_best,
                                            maxn = max_n_best,
                                            lr = 0.01,
                                            epoch = 10,
                                            minCount = 3,
                                            thread = 8,
                                            bucket=4000000)

model_subword.save_model(os.path.join(
                        out_path, '{}_d{}_w{}_m{}_M{}.bin'.format(
                        os.path.splitext(corpus)[0].split('/')[1], dim_best, ws_best, min_n_best, max_n_best)))
del model_subword

# train and save 2-5-gram model
model_big = fasttext.train_unsupervised(input = corpus,
                                            model = "skipgram",
                                            dim = dim_best,
                                            ws = ws_best,
                                            minn = 2,
                                            maxn = 5,
                                            lr = 0.01,
                                            epoch = 10,
                                            minCount = 3,
                                            thread = 8,
                                            bucket=4000000)

model_big.save_model(os.path.join(
                        out_path, '{}_d{}_w{}_m{}_M{}.bin'.format(
                        os.path.splitext(corpus)[0].split('/')[1], dim_best, ws_best, 2, 5)))
del model_big


# train and save bi-gram model
model_bigram = fasttext.train_unsupervised(input = corpus,
                                            model = "skipgram",
                                            dim = dim_best,
                                            ws = ws_best,
                                            minn = 2,
                                            maxn = 2,
                                            lr = 0.01,
                                            epoch = 10,
                                            minCount = 3,
                                            thread = 8,
                                            bucket=4000000)

model_bigram.save_model(os.path.join(
                        out_path, '{}_d{}_w{}_m{}_M{}.bin'.format(
                        os.path.splitext(corpus)[0].split('/')[1], dim_best, ws_best, 2, 2)))
del model_bigram