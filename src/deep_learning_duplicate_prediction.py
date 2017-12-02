"""Predicts Duplicate Questions in Quora using Deep learning."""
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.engine.topology import Merge
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text

data = pd.read_csv('train.csv')

data = data.dropna(axis=0)
y = data.is_duplicate.values

tk = text.Tokenizer(nb_words=200000)

max_len = 40

tk.fit_on_texts(list(data.question1.values.astype(str)) +
                list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values.astype(str))
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

word_index = tk.word_index

ytrain_enc = np_utils.to_categorical(y)

question_1 = Sequential()
question_1.add(Embedding(len(word_index) + 1,
                         300, input_length=40, dropout=0.2))
question_1.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

question_2 = Sequential()
question_2.add(Embedding(len(word_index) + 1,
                         300, input_length=40, dropout=0.2))
question_2.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

questions_combined = Sequential()
questions_combined.add(Merge([question_1, question_2], mode='concat'))
questions_combined.add(BatchNormalization())

questions_combined.add(Dense(300))
questions_combined.add(PReLU())
questions_combined.add(Dropout(0.2))
questions_combined.add(BatchNormalization())

questions_combined.add(Dense(300))
questions_combined.add(PReLU())
questions_combined.add(Dropout(0.2))
questions_combined.add(BatchNormalization())

questions_combined.add(Dense(300))
questions_combined.add(PReLU())
questions_combined.add(Dropout(0.2))
questions_combined.add(BatchNormalization())

questions_combined.add(Dense(300))
questions_combined.add(PReLU())
questions_combined.add(Dropout(0.2))
questions_combined.add(BatchNormalization())

questions_combined.add(Dense(300))
questions_combined.add(PReLU())
questions_combined.add(Dropout(0.2))
questions_combined.add(BatchNormalization())

questions_combined.add(Dense(1))
questions_combined.add(Activation('sigmoid'))

questions_combined.compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    'weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

questions_combined.fit([x1, x2], y=y, batch_size=384, nb_epoch=200,
                       verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])
