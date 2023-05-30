import fasttext
import os

## FIRST CREATE A CORRECT DATASET FROM THE DUTCH EMBEDDING EVAL JSON FILE

# Okay this is way more implementation than I initially thought, so I'm going to do this later


# let's first train two models (lexical and subword) with the best hyperparameters from the thesis
corpus = './processed_data/corpus/corpus_final.txt'
dim_best = 300
ws_best = 5
min_n_best = 2
max_n_best = 3
min_n_lexical = 0
max_n_lexical = 0
out_path = './processed_data/dsm/'

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