"""Utility functions for feature extraction."""

import re
import math
from collections import Counter


def cosine_similarity(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def jaccard_similarity(str_a, str_b):
    query = str_a.split(' ')
    document = str_b.split(' ')
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / float(len(union))


def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)


def lower_characters(text):
    return text.lower()


def simple_preprocess(text):
    text = lower_characters(text)
    text = remove_punctuations(text)
    return text


def remove_punctuations(text):
    return re.sub(r'[^\w\s]', '', text)
