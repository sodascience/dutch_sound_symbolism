import tarfile
import shutil
import gzip
import re
import os

def cgnl_extractor():
    '''
    :return:    none

    When called, this function opens the raw CGNL corpus, opens the ort.gz files 
    (i.e., the files that contain the raw text), removes lines that contain numbers
    or CGNL-specific annotations. The function then opens a .txt file and saves the 
    extracted text. Then, after all text is extracted, this function combines all .txt 
    files to create one complete CGNL corpus .txt file.
    '''

    ## CORPUS GESPROKEN NEDERLANDS (968,329 lines)

    # EXTRACT ALL TEXTFILES, DO SOME CGN-SPECIFIC CLEANING

    tar = tarfile.open('./raw_data/corpora/unextracted/corpus_gesproken_nederlands/CGNAnn2.0.3.tar.gz', 'r:gz')

    tarnames = tar.getnames()

    # Choose only the ortographic texts, and only the Dutch ones (leave out Flemish)
    ort_gzs = [n for n in tarnames if (n[:24] == 'Data/data/annot/text/ort' and n[32:34] == 'nl' and n[-7:] == '.ort.gz')]

    for name in ort_gzs:
        # Get the document name
        doc_name = name.split('/')[-1][:-7]

        # Extract ort.gz file
        temp_extract = tar.extract(name, path = './temp/')

        # Open the sub-tar file (every text is located within an individual sub-tar file)
        with gzip.open('./temp/' + name, 'rt', encoding='latin-1') as temp_file:
            doc = temp_file.read()
        
        # Assign an empty string to append cleaned lines to
        doc_cleaned = ''

        # For all lines in the text file
        for line in doc.splitlines():
            line = line.strip()
            # Skip line it contains numerics (all spoken numbers are written with text) and CGN-specific identifiers
            if re.search('1|2|3|4|5|6|7|8|9|0|<|=|IntervalTier|TextGrid', line):
                continue
            
            # Remove all parentheses and CGN-specific indicators of e.g. errors
            line = line.replace('"', '').replace("'", '').replace('.', '').replace('?', '').replace('xxx', '').replace('Xxx', '').replace('ggg', '')
            line = re.sub(r'\*.', '', line) 

            # Skip line if all that's left is fully uppercased (because then it's not spoken text but some other identifier)
            if line.isupper():
                continue
            elif line == "" or line == '""':
                continue
            else:
                doc_cleaned += line+'\n'

        # Open a .txt file and write the cleaned text to it
        with open('./raw_data/corpora/extracted/corpus_gesproken_nederlands/' + doc_name + '.txt', 'w') as f:
            f.write(doc_cleaned)              

        # Remove the temporary folder to save space on hard drive
        shutil.rmtree('./temp/')

    # COMBINE ALL TEXT FILES
    cgn_path = './raw_data/corpora/extracted/corpus_gesproken_nederlands/'
    with open(cgn_path + 'complete/cgn_complete.txt', 'a', encoding='utf-8') as cgn:
        for file in os.listdir(cgn_path):
            if os.path.isdir(os.path.join(cgn_path, file)):
                continue
            else:
                with open(cgn_path + file, 'r', encoding='utf-8') as temp_textfile:
                    for line in temp_textfile.readlines():
                        _ = cgn.write(line)
