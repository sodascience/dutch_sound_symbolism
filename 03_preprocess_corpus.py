'''
This script is used to extract text from the raw SoNaR-500 and CGNL data files using sonar500_exctractor() and cgnl_extractor().
Then, these corpora are combined with the 2018 CommonCrawl Dutch snapshot, after which the data is cleaned and stored as one
final corpus file. Cleaning steps taken include: remove non-alphabetic characters, split into sentences and store these on 
different lines, and lowercast all words.
'''

from src.preprocessing.sonar500 import sonar500_extractor
from src.preprocessing.combine import corpora_combiner
from src.preprocessing.test import pipeline_tester
from src.preprocessing.cgnl import cgnl_extractor
from src.preprocessing.corpus import make
from filesplit.split import Split

# RUN TO EXTRACT RELEVANT SONAR500 FILES FROM HUGE TARFILE
sonar500_extractor() 

# RUN TO EXTRACT RELEVANT CGNL FILES FROM TARFILE AND DO SOME MINOR CGNL-SPECIFIC CLEANING
cgnl_extractor()

# RUN TO COMBINE ALL TRHEE CORPORA TOGETHER (number of lines = 261,376,546)
corpora_combiner()

# RUN TO CREATE SMALL TEST CORPUS TO CHECK WHETHER MY PREPROCESSING PIPELINE WORKS WELL
pipeline_tester()

## SPLITTING MASTER FILE UP INTO CHUNCKS TO MAKE IT WORK TOGETHER WITH THE MAKE FUNCTION AND MULTIPROCESSING
split = Split(inputfile='./raw_data/corpora/extracted/complete_combined/corpora_complete_combined.txt',
              outputdir='./raw_data/corpora/extracted/complete_combined/corpus_chunks/').bylinecount(linecount = 270000)

## PREPROCESSING THE EXTRACTED SONAR500, CGN, and CC100-DUTCH
make(corpus_dir='./raw_data/corpora/extracted/complete_combined/corpus_chunks/', 
     out_path = './processed_data/corpus/corpus_final.txt',
     threads = 7)

