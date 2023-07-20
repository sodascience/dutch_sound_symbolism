'''
This script is used to generate BERT and fastText embeddings for the company names, first names, and nonwords 
that we gathered data for with our survey. Furthermore, the second half of this script combines the embeddings
for the three different types of names/words together into a unified dataframe, one for fastText and one for BERT. 
'''


from src.embeddings.embedding_fetchers import get_bert_embeddings, get_fasttext_embeddings, df_maker
import pandas as pd
import pickle

## LOAD DATA
# load the names/nonwords
company_data = pd.read_csv('./processed_data/words_and_names/survey_lists/company_names_final.csv')
fnames_data = pd.read_csv('./processed_data/words_and_names/survey_lists/dutch_names_final.csv')
nonword_data = pd.read_csv('./processed_data/words_and_names/survey_lists/nonwords_final.csv')

# lowercast the words
company_names = [word[0].lower() for word in company_data.values.tolist()]
first_names = [word[0].lower() for word in fnames_data.values.tolist()]
nonwords = [word[0].lower() for word in nonword_data.values.tolist()]


## BERT
# generate embeddings using the get_bert_embeddings() function
company_bert_embs = get_bert_embeddings(company_names)
fnames_bert_embs = get_bert_embeddings(first_names)
nonword_bert_embs = get_bert_embeddings(nonwords)

# save the embeddings
with open('./processed_data/embeddings/company_bert_embs.bin', 'wb') as f:
    pickle.dump(company_bert_embs, f)

with open('./processed_data/embeddings/fnames_bert_embs.bin', 'wb') as f:
    pickle.dump(fnames_bert_embs, f)

with open('./processed_data/embeddings/nonword_bert_embs.bin', 'wb') as f:
    pickle.dump(nonword_bert_embs, f)


## fastText
# generate embeddings using the get_fasttext_embeddings() function
company_ft_embs = get_fasttext_embeddings(company_names)
fnames_ft_embs = get_fasttext_embeddings(first_names)
nonword_ft_embs = get_fasttext_embeddings(nonwords)

# save the embeddings
with open('./processed_data/embeddings/company_ft_embs.bin', 'wb') as f:
    pickle.dump(company_ft_embs, f)

with open('./processed_data/embeddings/fnames_ft_embs.bin', 'wb') as f:
    pickle.dump(fnames_ft_embs, f)

with open('./processed_data/embeddings/nonword_ft_embs.bin', 'wb') as f:
    pickle.dump(nonword_ft_embs, f)

## Make DFs
from src.embeddings.embedding_fetchers import get_bert_embeddings, get_fasttext_embeddings, df_maker
import pandas as pd
import pickle

# open the embedding files
with open('./processed_data/embeddings/company_bert_embs.bin', 'rb') as f:
    company_bert_embs = pickle.load(f)
with open('./processed_data/embeddings/fnames_bert_embs.bin', 'rb') as f:
    fnames_bert_embs = pickle.load(f)
with open('./processed_data/embeddings/nonword_bert_embs.bin', 'rb') as f:
    nonword_bert_embs = pickle.load(f)

with open('./processed_data/embeddings/company_ft_embs.bin', 'rb') as f:
    company_ft_embs = pickle.load(f)
with open('./processed_data/embeddings/fnames_ft_embs.bin', 'rb') as f:
    fnames_ft_embs = pickle.load(f)
with open('./processed_data/embeddings/nonword_ft_embs.bin', 'rb') as f:
    nonword_ft_embs = pickle.load(f)

# create a dictionary with the embedding data that can be turned into a dataframe
survey_ft_embs = {'company' : company_ft_embs, 'fnames' : fnames_ft_embs, 'nonword' : nonword_ft_embs}
survey_bert_embs = {'company' : company_bert_embs, 'fnames' : fnames_bert_embs, 'nonword' : nonword_bert_embs}

# use df_maker to combine and save the dictionaries as a dataframe
survey_ft_df = df_maker(survey_ft_embs, 'survey_data', 'ft')
survey_bert_df = df_maker(survey_bert_embs, 'survey_data', 'bert')




