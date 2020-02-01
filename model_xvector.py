from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *

import os
import keras.backend as K
import keras_metrics as km

import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from analyzeWAV import CLASS_TYPE
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

CLASS_NUM = len(CLASS_TYPE)
DATA_DIRECT = './YoutubeAudio/NPY_DATA_segment/data/'
LABEL_DIRECT = './YoutubeAudio/NPY_DATA_segment/label/'

def mean_std_pool1d(x):
    mean = K.mean(x, axis=1)
    std = K.std(x, axis=1)
    return K.concatenate([mean, std], axis=1) 

def mean_std_pool1d_output_shape(input_shape):
    shape = list(input_shape)
    return tuple([shape[0], shape[2]*2])

def concat_npy(dir,itemlist):
    _res = []
    for item in tqdm(itemlist):
        _seg = np.load(os.path.join(dir,item))
        if len(_res)==0 :
            _res = _seg
        else:
            _res = np.concatenate([_res,_seg],axis=0)
    return _res

def train_test_split(csv='lib2.csv',ratio=0.1):
    df = pd.read_csv(csv)
    train_total = []
    test_total = []
    for target in CLASS_TYPE:
        classdf =df[df['aircraft_type']==target].reset_index(drop=True)
        n_train = int((1-ratio)*len(classdf))
        idxx = shuffle(np.arange(len(classdf)),random_state=12)
        id_train = idxx[0:n_train].tolist()
        id_test = idxx[n_train:].tolist()
        train_list = classdf.loc[id_train].Title.tolist()
        test_list = classdf.loc[id_test].Title.tolist()
        train_total.append(train_list)
        test_total.append(test_list)
    
    train_data_total = [item[:-4]+'_dt.npy' for sublist in train_total for item in sublist]
    train_label_total = [item[:-4]+'_lb.npy' for sublist in train_total for item in sublist]
    test_data_total = [item[:-4]+'_dt.npy' for sublist in test_total for item in sublist]
    test_label_total = [item[:-4]+'_lb.npy' for sublist in test_total for item in sublist]

    train_data_npy=concat_npy(DATA_DIRECT,train_data_total)
    train_label_npy=concat_npy(LABEL_DIRECT,train_label_total)
    test_data_npy=concat_npy(DATA_DIRECT,test_data_total)
    test_label_npy=concat_npy(LABEL_DIRECT,test_label_total)

    return train_data_npy,train_label_npy,test_data_npy,test_label_npy
        

def define_xvector():
    input_layer = Input(shape=(None, 140))
    # xvector model
    x = Conv1D(512, 3, padding='causal', name='tdnn_1')(input_layer)
    x = Activation('relu', name='tdnn_act_1')(x)
    x = Conv1D(512, 3, padding='causal', name='tdnn_2')(x)
    x = Activation('relu', name='tdnn_act_2')(x)
    x = Conv1D(512, 3, padding='causal', name='tdnn_3')(x)
    x = Activation('relu', name='tdnn_act_3')(x)
    x = Conv1D(512, 1, padding='causal', name='tdnn_4')(x)
    x = Activation('relu', name='tdnn_act_4')(x)
    x = Conv1D(1500, 1, padding='causal', name='tdnn_5')(x)
    x = Activation('relu', name='tdnn_act_5')(x)
    x = Lambda(mean_std_pool1d, output_shape=mean_std_pool1d_output_shape, name='stats_pool')(x)
    # fully-connected network
    x = Dense(512, activation='relu', name='feature_layer')(x)
    x = Dense(512, activation='relu', name='fc_2')(x)
    x = Dense(CLASS_NUM, activation='softmax', name='output_layer')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

def train_and_save_model(model='xvector',binary_class=False,single_class='glass'):
    model = define_xvector()
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['acc', km.precision(label=1), km.recall(label=0)])
    model.summary()
    callback_list = [
        ModelCheckpoint('checkpoint-{epoch:02d}.h5', monitor='loss', verbose=1, save_best_only=True, period=2), # do the check point each epoch, and save the best model
        ReduceLROnPlateau(monitor='loss', patience=3, verbose=1, min_lr=1e-6), # reducing the learning rate if the val_loss is not improving
        CSVLogger(filename='training_log.csv'), # logger to csv
        EarlyStopping(monitor='loss', patience=5) # early stop if there's no improvment of the loss
    ]
    tr_data,tr_label,ts_data,ts_label = train_test_split()
    encoder = LabelBinarizer()
    tr_label = encoder.fit_transform(tr_label)
    ts_label = encoder.transform(ts_label)
    print("Start Training process \nTraining data shape {} \nTraining label shape {}".format(tr_data.shape,tr_label.shape))
    model.fit(tr_data, tr_label, batch_size=16, epochs=100, verbose=1, validation_split=0.2)
    model.save('5class_segmentYoutube_model.h5')
    pred = model.predict(ts_data)
    pred = encoder.inverse_transform(pred)
    ts_label = encoder.inverse_transform(ts_label)
    cm = confusion_matrix(y_target=ts_label,y_predicted=pred,binary=False)
    cm = confusion_matrix(y_target=ts_label,y_predicted=pred,binary=False)
    plt.figure(figsize=(10,10))
    fig,ax = plot_confusion_matrix(conf_mat=cm)
    ax.set_xticklabels([''] + CLASS_TYPE,rotation = 40, ha='right')
    ax.set_yticklabels([''] + CLASS_TYPE)
    plt.savefig("ConfusionMatrix_segment_youtube.png")
    plt.show()
    # train_data, train_label = get_data(folder='training')
    # if binary_class == False:
    #     train_data, train_label = reduce_bg_data(train_data,train_label)
    # else:
    #     train_data,train_label = filter_reduce_bg(train_data,train_label,single_class)
    # encoder = LabelBinarizer()
    # train_label = encoder.fit_transform(train_label)
    # print("Start Training process \nTraining data shape {} \nTraining label shape {}".format(train_data.shape,train_label.shape))
    # model.fit(train_data, train_label, batch_size=32, epochs=25, verbose=1, validation_split=0.2)
    # if binary_class == False:
    #     model.save('re_{}_model.h5'.format(model))
    # else:
    #     model.save('re_{}_{}.h5'.format(model,single_class))

train_and_save_model()