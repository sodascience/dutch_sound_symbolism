from src.analyses.regression_analysis import fasttext_analysis, bert_analysis
import pickle

## LOAD DATA

with open('./processed_data/embeddings/company_ft_embs.bin', 'rb') as f:
    company_ft_embs = pickle.load(f)
with open('./processed_data/embeddings/fnames_ft_embs.bin', 'rb') as f:
    fnames_ft_embs = pickle.load(f)
with open('./processed_data/embeddings/nonword_ft_embs.bin', 'rb') as f:
    nonword_ft_embs = pickle.load(f)

with open('./processed_data/embeddings/company_bert_embs.bin', 'rb') as f:
    company_bert_embs = pickle.load(f)
with open('./processed_data/embeddings/fnames_bert_embs.bin', 'rb') as f:
    fnames_bert_embs = pickle.load(f)
with open('./processed_data/embeddings/nonword_bert_embs.bin', 'rb') as f:
    nonword_bert_embs = pickle.load(f)


## REGRESSION ANALYSES
LEXICAL = False
M = 2
RND_ITERS = 50
FEATURE = 'embedding'
TARGET = 'mean_rating'
GROUP = 'type'
ITEM = 'name'
FT_MODELS = ['0', '2', '2-3', '2-5']
BERT_LAYERS = list(range(12))
ATTRIBUTES = ['feminine', 'good', 'smart', 'trustworthy']

df_embeddings (columns = ['name', 'emb_model', 'embedding'])
df_ratings (columns = ['name', 'name_type', 'attribute', 'mean_rating'])


# FASTTEXT

company_ft_df, fnames_ft_df, nonword_ft_df = df_creator(rating_data, 
                                                        emb_type = 'ft', 
                                                        item = ITEM, 
                                                        feature = FEATURE, 
                                                        group = GROUP, 
                                                        attributes = ATTRIBUTES, 
                                                        target = TARGET, 
                                                        emb_model=None)




# BERT

company_bert_embs, fnames_bert_embs, nonword_bert_embs = fetch_bert_embeddings(bert_layer=None)