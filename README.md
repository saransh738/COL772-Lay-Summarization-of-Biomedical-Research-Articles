# Lay Summarization of Biomedical Research Articles

## Problem Statement:
Biomedical science has made many important discoveries, especially in healthcare. Most publications in this area use technical jargon understood only by area experts, doctors, and researchers. As a result, they are difficult for the general public to understand.
The goal is to use a pre-trained language model (PLM) and train a system that produces a layman’s summary given a research publication from the biomedical domain.

## Task Definition:
Given the abstract and main text of an article, the goal is to train a PLM-based model that generates a layman’s summary for the article. The task has two datasets - eLife and PLOS. An example from the dataset is given below.
````
# article lay summary
"lay_summary": "Messenger RNAs carry the...",

# article main text (abstract included), sections separated by "/n"
"article": "Gene expression varies widely ...",

# the article headings, corresponding to the sections in the text
"headings": ["Abstract", "Introduction", "Results", ..],

# keywords describing the topic of the article
"keywords": ["genetics", "biology", "genomics", ...],

# article id
"id": "journal.pgen.1002882",
````
## Approach:
We can model the task as seq2seq and fine-tune a PLM on the given datasets. We can start with small or base versions of the Flan-T5 and BioGPT. No other pre-trained
language models are allowed for fairness, and we will test you in a limited resource setting. We may also want to automatically decide which sections in the input article
to consider when producing the summary. We may explore domain ideas from the text summarization domain. For instance, we can use coverage loss to avoid focusing on the same section of the article. 
We can augment the available dataset using techniques like back-translation, self-training, and para-phrasing. We may further explore a multi-stage summarization
similar to Zhang et al. [2021] to handle large articles. Finally, we can consider using external knowledge sources that provide layman’s explanations for biomedical terms. 

## Evaluation Metric:
The model will be evaluated based on the following metrics
* Relevance: ROUGE (1, 2, and L) and BERTScore.
* Readability: Flesch-Kincaid Grade Level (FKGL)
* Factuality - AlignScore 

## Usage:
### Using run.sh
The command to train the model is
````
bash run_model.sh train <path_to_data> <path_to_save>
````
path to data is a directory containing all the train and validation JSONL files for both eLife and PLOS datasets. path to save is also a directory where the trained model must be saved.

The command to run inference using the model is
````
bash run_model.sh test <path_to_data> <path_to_model> <path_to_result>
````
path to data is a directory containing the two test files, path to model is a directory containing the model stored by the training command. Finally, path to result is a directory where the predicted summary files pols.txt and elife.txt must be stored.
