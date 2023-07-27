# Do Dutch Nonwords Convey Meaning?
## Analysis of Sound Symbolic Associations with Dutch Names and Nonwords using Distributional Semantic Models

Repository containing code and data used to compare word similarity from embeddings to human-rated word associations.

> Note: due to licensing and privacy issues, not all data could be provided in this repository. However, the corpora, open access data, and psycho_embeddings can be found online and downloaded to reproduce results.

This research uses fastText and BERT to find systematic relationships between (sub-)lexical word embeddings and human-
rated associations with names and words. Significant relationships based on sub-lexical patterns show that the form 
and meaning of a word are linked, and thus that meaning is signaled by how a word is spelled/sounds. 

This repository contains the code to train [fastText](https://fasttext.cc/) models and to compare the resulting models to human-rated associations. The BERT model used in this research was pretrained and can be found [here](https://huggingface.co/pdelobelle/robbert-v2-dutch-base). The ratings were obtained using a PsychoPy experiment which can be found [here](https://github.com/sodascience/word_norms_survey), and the association scores from this experiment are extracted using the code in [this repository](https://github.com/sodascience/bestworst_analysis).

<!-- Analyses -->
## Analyses
In this repository, two methods to quantify the relationship between language model embeddings and human association ratings are used. 

### Correlations
The first method is based on correlations and distance in the embedding space: first, a distance score is computed for each (non)word (e.g., "Annemarije") with a particular association (e.g., femininity). Then, the distance scores are correlated with our collected human association ratings for the same words. This is done for each association and language model separately.

This distance score is computed as follows: 
1. compute `similarity_left`: the average cosine similarity of the embedding of the target word with the embeddings of seed words related to one side of the semantic scale (e.g., female, feminine, grandma, aunt)
2. compute `similarity_right`: the average cosine similarity of the embedding of the target word with the embeddings of seed words related to the other side of the semantic scale (e.g., male, masculine, grandpa, uncle)
3. compute the `delta_score` as `similarity_right` - `similarity_left` 

For the fastText models, you can see the results [here](./results/analyses/correlation_analysis/correlations_ft0_&_2-5_bootstrap=False.csv). 

### Neural network regressions
In the second method, the raw embeddings for each word are entered as predictors in a neural network model with the human association rating as the target / outcome variable. Separate models are made for each association ("feminine", "good", "smart", "trustworthy") and language model. Hyperparameters (number of layers, number of hidden nodes per layer, learning rate, dropout rate, activation function) are tuned using grid search and 10-fold cross-validation.

For non-words, the better these predictions, the more evidence there is that the embeddings (and thus the word form) contains information about how these words relate to the associations investigated.

<!-- USAGE -->
## Usage

To reproduce our analyses, the scripts in this repository should be run in order. The scripts contain the following:

| Script                                                                                    | Contents                                                                            |
| :---------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| [`01_preprocessing.R`](./01_preprocessing.R)                                              | Data preparation of open access data                                                |
| [`02_data_visualization.R`](./02_data_visualization.R)                                    | Some visualisation and exploratory data analysis                                    |
| [`03_preprocess_corpus.py`](./03_preprocess_corpus.py)                                    | Extract, combine, and clean different raw corpus files                              |
| [`04_train_ft.py`](./04_train_ft.py)                                                      | Train various fasttext modlels using cleaned corpus                                 |
| [`05_get_word_embeddings.py`](./05_get_word_embeddings.py)                                | Create dfs with word embeddings                                                     |
| [`06_correlation_analysis.py`](./06_correlation_analysis.py)                              | Use word embeddings to calculate cosine scores with seed words                      |
| [`07_correlation_analysis.R`](./07_correlation_analysis.R)                                | Analyze the output files of script 06                                               |
| [`08_regression_analysis.py`](./08_regression_analysis.py)                                | Use word embeddings to do grid search and predict true association ratings using NNs|
| [`09_grid_search_analysis.R`](./09_grid_search_analysis.R)                                | Analyze grid search output generated under script 08                                |
| [`10_regression_analysis.R`](./10_regression_analysis.R)                                  | Analyze regression output generated under script 08                                 |
| [`11_external_data_correlation_analysis.py`](./11_external_data_correlation_analysis.py)  | Use open access data word embeddings to calculate cosine scores with seed words     |
| [`12_external_data_correlation_analysis.R`](./12_external_data_correlation_analysis.R)    | Analyze the output files of script 11                                               |


<!-- CONTACT -->
## Contact
This is a project by the [ODISSEI Social Data Science (SoDa)](https://odissei-data.nl/nl/soda/) team.
Do you have questions, suggestions, or remarks on the technical implementation? File an issue in the
issue tracker or feel free to contact [Aron Joosse](https://github.com/aron2vec), [Erik-Jan van Kesteren](https://github.com/vankesteren), or [Giovanni Cassani](https://github.com/GiovanniCassani).

<img src="docs/soda.png" alt="SoDa logo" width="250px"/> 