import numpy
numpy.random.seed(7)
import tensorflow as tf
tf.set_random_seed(7)

#import dataset
from keras.datasets import imdb

#import preprocessing library
from keras.preprocessing import sequence

#imports for the model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Embedding


results = []



def calc_model_score(max_features = 1000,
                     records_number = 25000,
                     max_review_length = 500,
                     embedding_vector_length = 64,
                     lstm_output_space = 64,
                     number_epochs = 50,
                     batch_size = 32):
    print ("Loading IMDB data...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features, seed = 113)
    print ("Train data set will be reduced to", records_number, "records...")
    X_train = X_train[:records_number]
    y_train = y_train[:records_number]
    print("Matching reviews length to ", max_review_length, "...")
    X_train = sequence.pad_sequences(X_train, maxlen = max_review_length, value=X_train[0][0])
    X_test = sequence.pad_sequences(X_test, maxlen = max_review_length, value=X_train[0][0])
    print("Creating model...")
    model = Sequential()
    model.add(Embedding(max_features, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(lstm_output_space))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print("Compiling model...")
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    print(model.summary())
    print("Fitting model...")
    history = model.fit(X_train, y_train, epochs=number_epochs, batch_size=batch_size)
    print("Calculating model accuracy...")
    model_score = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (model_score[1]*100))
    result_record = {"max_features":max_features,
                     "records_number":records_number,
                     "max_review_length":max_review_length,
                     "embedding_vector_length":embedding_vector_length,
                     "lstm_output_space":lstm_output_space,
                     "number_epochs":number_epochs,
                     "batch_size":batch_size,
                     "Accuracy": (model_score[1]*100),
                     "History":history}
    results.append(result_record)



calc_model_score()

import matplotlib.pyplot as plt


plt.figure(1)
plt.plot(results[0]['History'].history['acc'])
plt.figure(2)
plt.plot(results[0]['History'].history['loss'])
plt.figure(3)
plt.plot(results[0]['History'].history['acc'])
plt.plot(results[0]['History'].history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

