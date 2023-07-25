# Analysis of Sound Symbolic Associations with Dutch Names and Nonwords using Distributional Semantic Models

Repository containing code and data used to compare word similarity from embeddings to human-rated word associations.

> Note: due to licensing and privacy issues, not all data could be provided in this repository. However, the corpora, open access data, and psycho_embeddings can be found online and downloaded to reproduce results.

This research uses fastText and BERT to find systematic relationships between (sub-)lexical word embeddings and human-
rated associations with names and words. Significant relationships based on sub-lexical patterns show that the form 
and meaning of a word are linked, and thus that meaning is signaled by how a word is spelled/sounds. 

<!-- Analyses -->
## Analyses
In our repository, two methods to find relationships between embeddings and association ratings are used. 

The first method is unsupervised, using a list of seed words at two different ends of a semantic pole to which the cosine 
distance of the embedding of the target word is calculated. The target word's average proximity to both sides of the
semantic poles is used to create a delta score, the target word's place on the semantic scale. These delta scores 
are subsequently correlated with the association ratings.

The second method is supervised, where we train neural networks with a single linear activation node at the end that
predicts the association rating. Before making predictions, we run a grid search that tunes several hyperparameters of 
the neural networks.

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
issue tracker or feel free to contact [Erik-Jan van Kesteren](https://github.com/vankesteren).

<img src="docs/soda.png" alt="SoDa logo" width="250px"/> 