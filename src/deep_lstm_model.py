"""Model for identification of duplicate text."""
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Merge
from keras.layers.advanced_activations import PReLU


def get_questions_combined(INPUT_DIM, weights_path=None):
    question_1 = Sequential()
    question_1.add(Embedding(INPUT_DIM,
                             300, input_length=40, dropout=0.2))
    question_1.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

    question_2 = Sequential()
    question_2.add(Embedding(INPUT_DIM,
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

    if weights_path:
        questions_combined.load_weights(weights_path)

    return questions_combined
