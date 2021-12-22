import numpy as np
from collections import Counter
import librosa
import os
import multiprocessing
from sklearn.model_selection import train_test_split
from keras import utils
import keras
from keras.layers import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import sklearn
import math

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.metrics import classification_report

DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30  # Try with 20 also
EPOCHS = 10


def get_wav(file_name):
   
    y, sr = librosa.load('WavFormat/{}.wav'.format(file_name))
    print(file_name)
    return(librosa.core.resample(y=y, orig_sr=sr, target_sr=RATE, scale=True))


def to_mfcc(wav_array):
    
    return(librosa.feature.mfcc(y=wav_array, sr=RATE, n_mfcc=N_MFCC))


def split_people(df, test_size=0.2):
   
    return train_test_split(df['filename'], df['native_language'], test_size=test_size, random_state=1234)


def to_categorical(y):
   
    lang_dict = {}
    for index, language in enumerate(set(y)):
        lang_dict[language] = index
    y = list(map(lambda x: lang_dict[x], y))
    return utils.to_categorical(y, len(lang_dict))


def make_segments(mfccs, labels):
   
    segments = []
    seg_labels = []
    for mfcc, label in zip(mfccs, labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)


def cnn_model(X_train, y_train, X_validation, y_validation, batch_size=64):
  

    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    num_classes = len(y_train[0])
    input_shape = (rows, cols, 1)
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1)
    X_validation = X_validation.reshape(
        X_validation.shape[0], val_rows, val_cols, 1)

    model = keras.models.Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu',
               data_format="channels_last",
               input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='acc', min_delta=.005,
                       patience=10, verbose=1, mode='auto')
    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)
    # Image shifting
    datagen = ImageDataGenerator(width_shift_range=0.05)
    # Fit model using ImageDataGenerator
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=len(X_train) / 64, epochs=(EPOCHS+5),
                                  callbacks=[es, tb], validation_data=(X_validation, y_validation), verbose=2)

    #history = model.fit(X_train,y_train, batch_size = batch_size,epochs = EPOCHS,callbacks = [es,tb], validation_data=(X_validation,y_validation),verbose = 2)
    # pd.DataFrame(history.history).plot(figsize=(10, 5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    # plt.show()

    return (model)


def rf(X_train, y_train, X_validation, y_validation):
    df = pd.DataFrame(data=np.array(y_train), columns=[
                      "column1", "column2", "column3"])
    df['categorical'] = df[df.columns[1:]].apply(
        lambda x: ','.join(x.dropna().astype(int).astype(str)), axis=1)
    df.drop(['column1', 'column2', 'column3'], axis=1, inplace=True)
    df.loc[df['categorical'] == '0,0', :] = 0
    df.loc[df['categorical'] == '1,0', :] = 1
    df.loc[df['categorical'] == '0,1', :] = 2
    X_train1 = np.array(X_train)
    df1 = pd.DataFrame(X_train1.reshape(
        X_train1.shape[0], X_train1.shape[1] * X_train1.shape[2]))
    X_test1 = np.array(X_validation)
    df3 = pd.DataFrame(X_test1.reshape(
        X_test1.shape[0], X_test1.shape[1] * X_test1.shape[2]))
    df5 = pd.DataFrame(data=np.array(y_validation), columns=[
                       "column1", "column2", "column3"])
    df5['categorical'] = df5[df5.columns[1:]].apply(
        lambda x: ','.join(x.dropna().astype(int).astype(str)), axis=1)
    df5.drop(['column1', 'column2', 'column3'], axis=1, inplace=True)
    df5.loc[df5['categorical'] == '0,0', :] = 0
    df5.loc[df5['categorical'] == '1,0', :] = 1
    df5.loc[df5['categorical'] == '0,1', :] = 2
    rfc = RandomForestClassifier(n_estimators=150)
    rfc.fit(df1, df)
    y_pred = rfc.predict(df3)
    mse = sklearn.metrics.mean_squared_error(df5['categorical'], y_pred)
    rmse = math.sqrt(mse)
    print('Accuracy for Random Forest', 100*rmse)


def predict_class_audio(MFCCs, model):
    '''
    Predict class based on MFCC samples
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    MFCCs = MFCCs.reshape(MFCCs.shape[0], MFCCs.shape[1], MFCCs.shape[2], 1)
    y_predicted = model.predict_classes(MFCCs, verbose=0)
    return(Counter(list(y_predicted)).most_common(1)[0][0])


def predict_prob_class_audio(MFCCs, model):
    '''
    Predict class based on MFCC samples' probabilities
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    MFCCs = MFCCs.reshape(MFCCs.shape[0], MFCCs.shape[1], MFCCs.shape[2], 1)
    y_predicted = model.predict_proba(MFCCs, verbose=0)
    return(np.argmax(np.sum(y_predicted, axis=0)))


def predict_class_all(X_train, model):
    '''
    :param X_train: List of segmented mfccs
    :param model: trained model
    :return: list of predictions
    '''
    predictions = []
    for mfcc in X_train:
        predictions.append(predict_class_audio(mfcc, model))
        # predictions.append(predict_prob_class_audio(mfcc, model))
    return predictions


def confusion_matrix(y_predicted, y_test):
    '''
    Create confusion matrix
    :param y_predicted: list of predictions
    :param y_test: numpy array of shape (len(y_test), number of classes). 1.'s at index of actual, otherwise 0.
    :return: numpy array. confusion matrix
    '''
    confusion_matrix = np.zeros((len(y_test[0]), len(y_test[0])), dtype=int)
    for index, predicted in enumerate(y_predicted):
        confusion_matrix[np.argmax(y_test[index])][predicted] += 1
    return(confusion_matrix)


def get_accuracy(y_predicted, y_test):
    '''
    Get accuracy
    :param y_predicted: numpy array of predictions
    :param y_test: numpy array of actual
    :return: accuracy
    '''
    c_matrix = confusion_matrix(y_predicted, y_test)
    return(np.sum(c_matrix.diagonal()) / float(np.sum(c_matrix)))


def segment_one(mfcc):
    '''
    Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))


def create_segmented_mfccs(X_train):
    '''
    Creates segmented MFCCs from X_train
    :param X_train: list of MFCCs
    :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)


if __name__ == '__main__':
    df = pd.read_csv('Data/3_lang.csv')
    X_train, X_test, y_train, y_test = split_people(df)

    # Count of the training and testing samples
    train_count = Counter(y_train)
    test_count = Counter(y_test)

    # Gives 0.47440273037542663
    acc_to_beat = test_count.most_common(
        1)[0][1] / float(np.sum(list(test_count.values())))
    print(acc_to_beat)

    # Converting to numerical categories
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print("Loading .wav files!!!")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    X_train = pool.map(get_wav, X_train)
    X_test = pool.map(get_wav, X_test)
    print("Conversion to MFCC!!!")
    X_train = pool.map(to_mfcc, X_train)
    X_test = pool.map(to_mfcc, X_test)

    X_train, y_train = make_segments(X_train, y_train)
    X_validation, y_validation = make_segments(X_test, y_test)
    print("Data Ready!!")
    X_train, y_train = shuffle(X_train, y_train)

    # Randomizing training segments to create non-uniform data
    #X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)

    # model = cnn_model(np.array(X_train), np.array(y_train),np.array(X_validation), np.array(y_validation))

    # y_predicted = predict_class_all(create_segmented_mfccs(X_test), model)
    rf(X_train, y_train,X_validation,y_validation)

    # Print statistics
    print('Training samples:', train_count)
    print('Testing samples:', test_count)
    #print('Accuracy to beat:', acc_to_beat)
    print('Confusion matrix of total samples:\n', np.sum(
        confusion_matrix(y_predicted, y_test), axis=1))
    print('Confusion matrix:\n', confusion_matrix(y_predicted, y_test))
    print('Accuracy:', get_accuracy(y_predicted, y_test))


# Training samples: Counter({'english': 508, 'spanish': 189, 'arabic': 157})
# Testing samples: Counter({'english': 138, 'spanish': 39, 'arabic': 37})
# Confusion matrix of total samples:
#  [ 39 138  37]
# Confusion matrix:
#  [[  9  22   8]
#  [  2 132   4]
#  [  2  13  22]]
# Accuracy: 0.7616822429906542
