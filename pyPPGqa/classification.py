"""
Functions to process model inputs and outputs.

This module provides functions that classify 5 minutes PPG time-series or PPG images 
into Reliable or Unreliable for each HR-HRV features.

Copyright 2020, Emad Kasaeyan Naeini
Licence: MIT, see LICENCE for more details.

"""

from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime
import time, timeit
import heartpy as hp
import pandas as pd
import numpy as np
import os, sys, glob, pickle, tempfile
from PIL import  Image
from sklearn.metrics import classification_report, confusion_matrix,roc_curve, auc
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras import applications
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

BATCH_SIZE = 1024
EPOCHS = 15
def rgba2rgb():
    frames_dir = './data/frames/'
    frames = sorted(glob.glob(frames_dir+'/*'))
    X_train = np.zeros((len(frames),224,224,3), dtype='uint8')
    for ind in range(len(frames)):
        png = Image.open(frames[ind])
        png.load() # required for png.split()
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
        img = np.asarray(background)
        X_train[ind] = img
    np.save('frames.npy', X_train)
    return

# # Dataset Prepartion
def load_1Ddataset(feature):
    train_data = pd.read_csv('data/train.csv', header=None).to_numpy()
    train_label = pd.read_csv('data/{}_label.csv'.format(feature), header=None).to_numpy()
    train_data = train_data[:,:,np.newaxis]
    neg, pos = np.bincount(train_label.ravel())
    total = neg + pos
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    initial_bias = np.log([pos/neg])
    trX, valX, trY, valY = train_test_split(train_data, train_label, test_size=0.2, random_state=42)
    # one hot encode y
    trY = to_categorical(trY)
    valY = to_categorical(valY)
    return trX, valX, trY, valY, initial_bias, class_weight

# # Dataset Prepartion
def load_2Ddataset(feature):

    # train_data = rgba2rgb()
    train_data = np.load('frames.npy')
    train_data = train_data / 255.0
    train_label = pd.read_csv('data/{}_label.csv'.format(feature), header=None).to_numpy()
    neg, pos = np.bincount(train_label.ravel())
    total = neg + pos
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    initial_bias = np.log([pos/neg])
    trX, valX, trY, valY = train_test_split(train_data, train_label, test_size=0.2, random_state=42)
    # one hot encode y
    trY = to_categorical(trY)
    valY = to_categorical(valY)
    return trX, valX, trY, valY, initial_bias, class_weight

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def create_1Dcnn(n_filters=32, output_bias=None):
    # if using tensorflow 2 you can substitute the following metrics with the current ones
    # METRICS = [
    #       keras.metrics.TruePositives(name='tp'),
    #       keras.metrics.FalsePositives(name='fp'),
    #       keras.metrics.TrueNegatives(name='tn'),
    #       keras.metrics.FalseNegatives(name='fn'), 
    #       keras.metrics.BinaryAccuracy(name='accuracy'),
    #       keras.metrics.Precision(name='precision'),
    #       keras.metrics.Recall(name='recall'),
    #       keras.metrics.AUC(name='auc'),
    # ]
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = models.Sequential()
    model.add(layers.Conv1D(filters=n_filters, kernel_initializer='he_normal', kernel_size=3, activation='relu', input_shape=(60*20,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=n_filters*2, kernel_initializer='he_normal', kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(348, kernel_initializer='he_normal', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2,activation='sigmoid', bias_initializer=output_bias))
    # myAdam = tf.keras.optimizers.Adam(lr=0.00001)
    myAdam = Adam(lr=0.00001)
    model.compile(optimizer=myAdam,
                  loss='binary_crossentropy',
                  metrics=['accuracy', recall_m, precision_m, f1_m])
    return model

def create_2Dcnn(model_name, output_bias=None):
    # if using tensorflow 2 you can substitute the following metrics with the current ones
    # METRICS = [
    #       keras.metrics.TruePositives(name='tp'),
    #       keras.metrics.FalsePositives(name='fp'),
    #       keras.metrics.TrueNegatives(name='tn'),
    #       keras.metrics.FalseNegatives(name='fn'), 
    #       keras.metrics.BinaryAccuracy(name='accuracy'),
    #       keras.metrics.Precision(name='precision'),
    #       keras.metrics.Recall(name='recall'),
    #       keras.metrics.AUC(name='auc'),
    # ]
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    if model_name=='VGG16':
        base_model = applications.vgg16.VGG16(weights='imagenet', include_top=False)
    elif model_name=='ResNet50':
        base_model = keras.applications.keras_applications.resnet.ResNet50(weights='imagenet', include_top=False, backend=keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    elif model_name=='MobileNetV2':
        base_model = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, backend=keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)

    for layer in base_model.layers:
        layer.trainable = False

    input = Input(shape=(224,224,3))
    # x = base_model.output
    x = base_model(input)
    x = Flatten()(x)

    # x = GlobalMaxPooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(2, activation='sigmoid', bias_initializer=output_bias)(x)
    model = Model(inputs=input, outputs=predictions)
    myAdam = Adam(lr=0.00005)
    model.compile(optimizer=myAdam,
                loss='binary_crossentropy',
                metrics=['accuracy', recall_m, precision_m, f1_m])
    return model

# # Fit and Evaluate Model
def evaluate_1Dmodel(trX, valX, trY, valY, initial_bias, class_weight, feat, model_name, load=False):
    verbose, epochs, batch_size = 2, 15, 1024
    # Prepare callbacks for model saving and for learning rate adjustment.
    filepath = save_model(feat, model_name=model_name)
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_auc',
                                 verbose=1,
                                 save_best_only=True)
    callbacks = [checkpoint]
    model = create_1Dcnn(n_filters=32, output_bias=initial_bias)
    if load:
        filepath = load_model(feat, model_name)
        model.load_weights(filepath)
        # fit network
        history = model.fit(trX, trY, epochs=epochs, validation_data=(valX,valY), batch_size=batch_size, verbose=verbose, callbacks=callbacks, class_weight=class_weight)
        with open('./saved_results/{}_{}_trainHistoryDict'.format(feat, model_name), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    else:
        # fit network
        history = model.fit(trX, trY, epochs=epochs, validation_data=(valX,valY), batch_size=batch_size, verbose=verbose, callbacks=callbacks, class_weight=class_weight)
        with open('./saved_results/{}_{}_trainHistoryDict'.format(feat, model_name), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    train_pred = model.predict(trX, batch_size=1024)
    test_pred = model.predict(valX, batch_size=1024)
    # evaluate model
    results = model.evaluate(valX, valY, batch_size=batch_size, verbose=0)
    return model.metrics_names, history, results, train_pred, test_pred

def evaluate_2Dmodel(trX, valX, trY, valY, initial_bias, class_weight, feat, model_name, load=False):
    verbose, epochs, batch_size = 2, 15, 1024
    # Prepare callbacks for model saving and for learning rate adjustment.
    filepath = save_model(feat, model_name)
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_auc',
                                 verbose=2,
                                 save_best_only=True)
    callbacks = [checkpoint]
    model = create_2Dcnn(model_name, output_bias=initial_bias)
    if load:
        filepath = load_model(feat, model_name)
        model.load_weights(filepath)
        # fit network
        history = model.fit(trX, trY, epochs=epochs, validation_data=(valX,valY), batch_size=batch_size, verbose=verbose, callbacks=callbacks, class_weight=class_weight)
        with open('./saved_results/{}_{}_trainHistoryDict'.format(feat, model_name), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    else:
        # fit network
        history = model.fit(trX, trY, epochs=epochs, validation_data=(valX,valY), batch_size=batch_size, verbose=verbose, callbacks=callbacks, class_weight=class_weight)
        with open('./saved_results/{}_{}_trainHistoryDict'.format(feat, model_name), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    train_pred = model.predict(trX, batch_size=1024)
    test_pred = model.predict(valX, batch_size=1024)
    # evaluate model
    results = model.evaluate(valX, valY, batch_size=batch_size, verbose=0)
    return model.metrics_names, history, results, train_pred, test_pred

def summarize_results(names, scores, feature, model_names):
    df = pd.DataFrame(scores, index=model_names, columns=names)
    df.to_csv('./saved_results/{}_train_results.csv'.format(feature))

# Run the experiments
def run_experiments(load, features, model_names):

    for feat in features:
        # Evaluate each feature
        all_scores = list()
        for model_name in model_names:
            start = time.perf_counter()
            print(model_name, feat)
            if model_name == '1DCNN':
                trainX, testX, trainy, testy, initial_bias, class_weight = load_1Ddataset(feat)
                names, history, results, train_pred, test_pred = evaluate_1Dmodel(trainX, testX, trainy, testy, initial_bias, class_weight, feat=feat, model_name=model_name, load=load)
            else:
                trainX, testX, trainy, testy, initial_bias, class_weight = load_2Ddataset(feat)
                names, history, results, train_pred, test_pred = evaluate_2Dmodel(trainX, testX, trainy, testy, initial_bias, class_weight, feat=feat, model_name=model_name, load=load)
          
            all_scores.append(results)
            print('\nModel %s Feature %s in %.8s sec' %( model_name, feat, (time.perf_counter()-start)))
        summarize_results(names, all_scores, feature, model_names)
        
# Prepare model saving directory.
def save_model(feat, model_name):
    save_dir = os.path.join(os.getcwd(), 'saved_models/'+feat)
    model_name = 'ppg_confidence_%s_%s.{epoch:03d}.h5' % (feat,model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    return filepath

def load_model(feat, model_name):
    load_dir = os.path.join(os.getcwd(), 'saved_models/'+feat)
    model_weights = glob.glob(load_dir+'/*')
    for model in model_weights:
        if model.split('_')[-1].split('.')[0] == model_name:
            filepath = model
    return filepath

if __name__ == "__main__":
    features = ['hr', 'avnn', 'rmssd', 'sdnn']
    model_names = ['1DCNN', 'MobileNetV2', 'VGG16', 'ResNet50']
    
    rgba2rgb()

    load = True
    save_dir = './saved_results'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    run_experiments(load, features, model_names)
