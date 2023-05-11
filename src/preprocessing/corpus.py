from datetime import datetime
import multiprocessing as mp
import spacy
import re
import os

def _mp_clean_text(args):

    return clean_text(args)

def clean_text(in_path):
    nlp = spacy.load('nl_core_news_md')
    nlp.max_length = 10000000000

    sentence_list = []

    with nlp.select_pipes(disable=["lemmatizer", "tok2vec", "tagger", "parser"]):
        nlp.enable_pipe("senter")  ## Helps with better segmenting into sentences
        with open(in_path, 'r') as f_in:
            for line in f_in:
                line = re.sub("<[A-Za-z]+>", ".", line)
                
                doc = nlp(line)
                sentence = []

                for token in doc:
                    if token.is_sent_start:
                        if not sentence:
                            pass
                        else:
                            sentence_list.append(sentence)
                            sentence = []

                    if token.is_upper is True:                  # Remove all full-caps words
                        continue
                    # elif token.text.lower() in filter:          # Remove all words that are in the list of banned words
                    #     continue
                    elif token.is_alpha:
                        sentence.append(token.text.lower())

    
    return sentence_list

def make(corpus_dir, out_path, threads=32):

    files = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, f))]

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Started processing {} corpus files...".format(len(files))))
    
    with open(out_path, 'w') as f_out:
        with mp.Pool(threads) as pool:
            outputs = pool.imap(_mp_clean_text, ((f) for f in files))
            for output in outputs:
                for sentence in output:
                    f_out.write(' '.join(sentence))
                    f_out.write('\n')
                    
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done."))