"""Similarity metrics from text to identify duplicate questions."""
import pandas as pd
from fuzzywuzzy import fuzz
from .utils import cosine_similarity, jaccard_similarity, simple_preprocess


df_train = pd.read_csv('train.csv').fillna("")

# Ratio's based on fuzzywuzzy library
df_train['fuzz_ratio'] = df_train.apply(
    lambda row: fuzz.ratio(row['question1'], row['question2']), axis=1)
df_train['partial_ratio'] = df_train.apply(
    lambda row: fuzz.partial_ratio(row['question1'], row['question2']), axis=1)
df_train['token_sort_ratio'] = df_train.apply(
    lambda row: fuzz.token_sort_ratio(row['question1'], row['question2']), axis=1)
df_train['token_set_ratio'] = df_train.apply(
    lambda row: fuzz.token_set_ratio(row['question1'], row['question2']), axis=1)

# Jaccard and Cosine Similarity metrics on unigrams
df_train['jaccard_similarity'] = df_train.apply(
    lambda row: jaccard_similarity(row['question1'], row['question2']), axis=1)
df_train['cosine_similarity'] = df_train.apply(
    lambda row: cosine_similarity(row['question1'], row['question2']), axis=1)

# Remove Punctuations and clean the text
df_train['ques1_clean'] = df_train.apply(
    lambda row: simple_preprocess(row['question1']), axis=1)
df_train['ques2_clean'] = df_train.apply(
    lambda row: simple_preprocess(row['question2']), axis=1)

# Jaccard and Cosine Similarity metrics on Cleaned text (unigram)
df_train['jaccard_similarity_cleaned'] = df_train.apply(
    lambda row: jaccard_similarity(row['ques1_clean'], row['ques2_clean']), axis=1)
df_train['cosine_similarity_cleaned'] = df_train.apply(
    lambda row: cosine_similarity(row['ques1_clean'], row['ques2_clean']), axis=1)

df_train.to_csv('Training_data_with_similarity_features.csv', index=False)
