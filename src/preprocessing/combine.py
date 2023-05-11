import shutil

def corpora_combiner():
    snr_path = './raw_data/corpora/extracted/sonar500/complete/sonar500_complete.txt'
    cgn_path = './raw_data/corpora/extracted/corpus_gesproken_nederlands/complete/cgn_complete.txt'
    cc100_path = './raw_data/corpora/extracted/cc100_dutch/nl.txt'

    with open('./raw_data/corpora/extracted/complete_combined/corpora_complete_combined.txt','wb') as corpus_combined:
        for f in [snr_path, cgn_path, cc100_path]:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, corpus_combined)