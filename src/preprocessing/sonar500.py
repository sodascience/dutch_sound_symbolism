import folia.main as folia
import tarfile
import shutil
import os

def sonar500_extractor():
    '''
    :return:    none

    When called, this function opens the raw SoNaR-500 corpus, opens the folia.xml files 
    (i.e., the files that contain the raw text), then opens a .txt file and saves the 
    extracted text. Then, after all text is extracted, this function combines all .txt 
    files to create one complete SoNaR corpus .txt file.
    '''

    ## SONAR-500 CORPUS (30,494,087 lines)

    # OPEN AND BROWSE TAR FILE
    tar = tarfile.open('./raw_data/corpora/unextracted/sonar_500/20150602_SoNaRCorpus_NC_1.2.1.tgz', 'r:gz')

    tarnames = tar.getnames()

    # Choose the files that have the information we need (folia files)
    folia_xmls = [n for n in tarnames if "folia.xml" in n]

    for name in folia_xmls:
        # Get the document name
        doc_name = name.split('/')[-1]

        # Extract the .folia.xml file to a temporary folder so we can open it with folia.Document
        temp_file = tar.extract(name, path = './temp/')

        # Try opening the file, if it's corrupted or anything, just continue onward
        try: 
            doc = folia.Document(file='./temp' + name)
        except Exception: 
            # Something went wrong while loading WR-P-E-A-0000303138.folia.xml
            print("Something went wrong while loading " + doc_name)
            continue 

        # Open a .txt file and save the unannotated text from the folia.xml file 
        with open('./raw_data/corpora/extracted/sonar500/' + doc_name[:-10] + '.txt', 'w') as f:
            f.write(doc.text())

        # Remove the temporary folder to save space on hard drive
        shutil.rmtree('./temp/')

    # COMBINE ALL EXTRACTED TEXT FILES
    snr_path = './raw_data/corpora/extracted/sonar500/'
    with open(snr_path + 'complete/sonar500_complete.txt', 'a', encoding='utf-8') as snr500:
        for file in os.listdir(snr_path):
            if os.path.isdir(os.path.join(snr_path + file)):
                continue
            else:
                with open(snr_path + file, 'r', encoding='utf-8') as temp_textfile:
                    for line in temp_textfile.readlines():
                        _ = snr500.write(line)

