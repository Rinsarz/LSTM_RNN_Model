import nltk
from nltk import word_tokenize
from nltk.util import ngrams

import numpy
import random
import tensorflow as tf
from keras.preprocessing import sequence

numpy.random.seed(7)
tf.set_random_seed(7)

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Embedding

# Just to ensure
numpy.random.seed(7)
tf.set_random_seed(7)


def convert_to_trigrams(in_set, out_set):
    new_in = []
    new_out = []
    for x, y in zip(in_set, out_set):
        tg = list(ngrams(x, 3))
        nx = []
        for t in tg:
            if t in tg_dict:
                nx.append(tg_dict[t])
        if len(nx) > 0:
            new_in.append(nx)
            new_out.append(int(y))
    return new_in, new_out

def convert_to_trigrams_and_bigrams(in_set, out_set):
    new_in = []
    new_out = []
    for x, y in zip(in_set, out_set):
        tg = list(ngrams(x, 3))
        tg.extend(list(ngrams(x, 2)))
        nx = []
        for t in tg:
            if t in tg_dict:
                nx.append(tg_dict[t])
        if len(nx) > 0:
            new_in.append(nx)
            new_out.append(int(y))
    return new_in, new_out


def fitLength(arr, m_length):
    for i in range(len(arr)):
        l = len(arr[i]) - 1
        while len(arr[i]) < m_length:
            arr[i].append(arr[i][random.randint(0, l)])  # different elements can be used to increase length
        if len(arr[i]) > m_length:
            arr[i] = arr[i][:m_length]  # here some different methods can be applied (sorting, etc.)


print("Loading IMDB data...")
max_features = 1000
(X_train, y_train), (X_test, y_test) = imdb.load_data(seed=113)
print("IMDB data loaded.")

print("Creating trigrams bow...")
trigrams_bow = []
for x in X_train:
    trigrams_bow.extend(list(ngrams(x, 3)))

print("Creating bigrams bow...")
bigrams_bow = []
for x in X_train:
    bigrams_bow.extend(list(ngrams(x, 2)))

print("Creating trigram bow finished.")
print("Calculating frequency distribution for trigrams...")
fdist_tri = nltk.FreqDist(trigrams_bow)
len(fdist_tri)

print("Calculating frequency distribution for bigrams...")
fdist_bi = nltk.FreqDist(bigrams_bow)
len(fdist_bi)

print("Creating trigrams dictionary...")
tg_dict = {}
dict_size = 500
count = 0
for tg, freq in fdist_tri.most_common(dict_size):
    tg_dict[tg] = count
    count += 1

for tg, freq in fdist_bi.most_common(dict_size):
    tg_dict[tg] = count
    count += 1

print("Dictionary created.")

print("Converting IMDB data to trigrams...")
x_train_tg, y_train_tg = convert_to_trigrams_and_bigrams(X_train, y_train)
x_test_tg, y_test_tg = convert_to_trigrams_and_bigrams(X_test, y_test)
print("IMDB converted.")

print("Reviews length distribution:")
import matplotlib.pyplot as plt

review_length = []
for x in range(0, len(x_train_tg)):
    review_length.append(len(x_train_tg[x]))

plt.figure(1)
plt.plot(review_length)

plt.figure(2)
plt.plot(review_length)
plt.axis([0, 5000, 0, 200])

plt.show()

max_review_length = 300
print("Matching reviews length to ", max_review_length, "...")
# x_train_tg = sequence.pad_sequences(x_train_tg, maxlen = max_review_length, value=x_train_tg[0][0])
# x_test_tg = sequence.pad_sequences(x_test_tg, maxlen = max_review_length, value=x_test_tg[0][0])

fitLength(x_train_tg, max_review_length)
fitLength(x_test_tg, max_review_length)
print("Matching finished.")

print("Creating model...")
model = Sequential()
embedding_vector_length = 32
lstm_output_space = 50  # parameter to change
model.add(Embedding(max_features, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(lstm_output_space, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print("Compiling model...")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print("Fitting model...")
history = model.fit(x_train_tg, y_train_tg, epochs=15, batch_size=32)

print("Calculating model accuracy...")
model_score = model.evaluate(x_test_tg, y_test_tg, verbose=0)
print("Accuracy: %.2f%%" % (model_score[1] * 100))