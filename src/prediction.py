"""Given any two text, tell whether they are duplicates or not."""
import pandas as pd
from keras.preprocessing import sequence, text
from .deep_lstm_model import get_questions_combined

data = pd.read_csv('train.csv')
data = data.dropna(axis=0)
y = data.is_duplicate.values
tk = text.Tokenizer(nb_words=200000)
max_len = 40

tk.fit_on_texts(list(data.question1.values.astype(str)) +
                list(data.question2.values.astype(str)))
word_index = tk.word_index
INPUT_DIM = len(word_index) + 1

x1 = tk.texts_to_sequences(data.question1.values.astype(str))
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

questions_combined = get_questions_combined(
    INPUT_DIM=INPUT_DIM, weights_path='weights.h5')
questions_combined.compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

questions_combined.predict([x1, x2])
