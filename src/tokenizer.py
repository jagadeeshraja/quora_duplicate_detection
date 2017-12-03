"""Train, load and save tokenizer."""
import pandas as pd
from keras.preprocessing import text


def load_tokenizer(pickle_file=None):
    if not pickle_file:
        data = pd.read_csv('train.csv')
        data = data.dropna(axis=0)
        tk = text.Tokenizer(nb_words=200000)
        tk.fit_on_texts(list(data.question1.values.astype(str)) +
                        list(data.question2.values.astype(str)))
        return tk
