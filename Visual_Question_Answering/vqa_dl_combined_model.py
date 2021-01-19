#Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation,Flatten,BatchNormalization
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate, multiply
from keras.utils import np_utils, plot_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

#load train datafiles
def load_train_datafiles():
    ques_train= np.load('ques_train.npy')
    cnn_train=np.load('cnn_train.npy')
    rpn_train=np.load('rpn_train.npy')
    ans_encoded_train=np.load('ans_encoded_train.npy')
    embedding_matrix=np.load('embedding_matrix.npy')
    return ques_train,cnn_train,rpn_train,ans_encoded_train,embedding_matrix

#load validation datafiles
def load_val_datafiles():
    ques_val= np.load('ques_val.npy')
    cnn_val=np.load('cnn_val.npy')
    rpn_val=np.load('rpn_val.npy')
    ans_encoded_val=np.load('ans_encoded_val.npy')
    return ques_val,cnn_val,rpn_val,ans_encoded_val

#define combined model definition
def VQA_Model_definition(vocab_size,emb_length,max_input_length):
    #IMAGE MODEL
    cnn_image_Input = Input(shape=(2048,))
    model_cnn_image = Dense(2048, activation='tanh')(cnn_image_Input)
    model_cnn_image = BatchNormalization()(model_cnn_image)

    #RPN IMAGE MODEL
    rpn_image_Input = Input(shape=(1444,))
    model_rpn_image = Dense(2048, activation='tanh')(rpn_image_Input)
    model_rpn_image = BatchNormalization()(model_rpn_image)

    #LANGUAGE MODEL
    language_input = Input(shape=(max_input_length,))
    model_language = Embedding(vocab_size, emb_length, input_length=max_input_length)(language_input)
    model_language = Flatten()(model_language)
    model_language = Dense(2048, activation='tanh')(model_language)
    model_language = BatchNormalization()(model_language)

    # COMBINED MODEL
    merge = multiply([model_language, model_cnn_image, model_rpn_image])
    output = Dense(2048, activation='tanh')(merge)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(1024, activation='tanh')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(1024, activation='tanh')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(1001,activation='softmax')(output)

    VQA_model = Model(inputs=[language_input,cnn_image_Input,rpn_image_Input], outputs=output)

    return VQA_model



if __name__ == '__main__':
    #load train files
    ques_train,cnn_train,rpn_train,ans_encoded_train,embedding_matrix = load_train_datafiles()
    #load validation files
    ques_val,cnn_val,rpn_val,ans_encoded_val = load_val_datafiles()
    vocab_size, emb_length = embedding_matrix.shape
    max_input_length = 26

    #load vqa model definition
    VQA_Model = VQA_Model_definition(vocab_size, emb_length, max_input_length)

    #combine datafiles in the format of train and validation input
    train_X = [ques_train, cnn_train, rpn_train]
    train_y = ans_encoded_train

    val_X = [ques_val, cnn_val, rpn_val]
    val_y = ans_encoded_val

    #define checkpoint path to save best model
    filepath="weights.best_full_BN_drop.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    optimizer = keras.optimizers.adam(learning_rate=0.0005)
    VQA_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    VQA_model.summary()

    #start model training
    model_train = VQA_model.fit(train_X, train_y, batch_size = 64, epochs=1000, validation_data=(val_X, val_y), shuffle=True, callbacks=callbacks_list)
