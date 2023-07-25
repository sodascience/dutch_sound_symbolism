from src.preprocessing.corpus import make
from filesplit.split import Split

def pipeline_tester():
    '''
    :return:    none

    When called, this function tests on a small portion of the full corpus whether the
    corpus pre-processing and cleaning pipeline works and produces the intended results.
    '''

    with open('./raw_data/corpora/extracted/complete_combined/corpora_complete_combined.txt','r') as corpus_raw:
        with open('./raw_data/corpora/extracted/complete_combined/test_corpus.txt','a') as corpus_test:
            for i in range(1000):
                corpus_test.write(corpus_raw.readline())

    split = Split(inputfile='./raw_data/corpora/extracted/complete_combined/test_corpus.txt',
                outputdir='./raw_data/corpora/extracted/complete_combined/test_corpus/').bylinecount(linecount = 100)

    make(corpus_dir='./raw_data/corpora/extracted/complete_combined/test_corpus/', 
        out_path = './processed_data/corpus/test_corpus.txt',
        threads = 7)