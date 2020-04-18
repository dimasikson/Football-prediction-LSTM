
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
from datetime import datetime
from utils import loadData
import time
import ast
# from modules import C_Concat

leagues = [
    'E0'
    # 'D1'
    # 'I1'
    # 'SP1'
    # 'F1'
    # 'N1'
    # 'P1'
    # 'T1'
]

modelName =  leagues[0] + '-footballPrediction-' + str(int(time.time()))

tensorboard = TensorBoard(log_dir='.\logs\{}'.format(modelName))

def buildModel(inputs1, inputs2=None, lstm=True, lossType='categorical', dropout=True, denseActivation='relu', units=32, lr=1e-3, decay=1e-5):

    if lossType == 'goaldif':
        denseActivation=None

    i = 1

    if lstm==True:
        inputDim1 = inputs1.shape[1]
        inputDim2r = inputs2[0].shape[0]
        inputDim2c = inputs2[0].shape[1]

        inpt1 = tf.keras.Input(shape=(inputDim1,))
        x1 = tf.keras.layers.Dense(units=units, activation=denseActivation, name=str(units)+'dense'+str(i))(inpt1);i+=1

        inpt2 = tf.keras.Input(shape=(inputDim2r,inputDim2c))
        x2 = tf.keras.layers.LSTM(units, name=str(units)+'lstm'+str(i))(inpt2);i+=1
        if dropout:
            x2 = tf.keras.layers.Dropout(0.1)(x2)

        x2 = tf.keras.layers.Dense(units=units, activation=denseActivation, name=str(units)+'dense'+str(i))(x2);i+=1

        x = tf.keras.layers.concatenate([x1, x2], axis=1)

    else:
        inputDim = inputs1.shape[1]

        inpt = tf.keras.Input(shape=(inputDim,))
        x = tf.keras.layers.Dense(units=units, activation=denseActivation, name=str(units)+'dense'+str(i))(inpt);i+=1

    for j in range(5,2,-1):
        for _ in range(3):
            x = tf.keras.layers.Dense(units=2**j, activation=denseActivation, name=str(2**j)+'dense'+str(i))(x);i+=1
            # x = tf.keras.layers.Dropout(0.1)(x)

    if lossType == 'return':
        outActivation='relu'
        outUnits=3
    elif lossType == 'goals':
        outActivation='relu'
        outUnits=2
    elif lossType == 'goaldif':
        outActivation='tanh'
        outUnits=1
    else:
        outActivation='softmax'
        outUnits=3

    out = tf.keras.layers.Dense(units=outUnits, activation=outActivation, name=str(2**i)+'denseOutput'+str(i))(x);i+=1

    if lstm==True:
        inpts = [inpt1, inpt2]
    else:
        inpts = [inpt]

    model = tf.keras.Model(inputs=inpts, outputs=[out])

    if lossType in ['return', 'goals', 'goaldif']:
        modelLoss = tf.keras.losses.MeanSquaredError()
        model.compile(
            loss=modelLoss,
            optimizer=tf.keras.optimizers.Adam(lr=lr, decay=decay)
        )
    else:
        modelLoss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(
            loss=modelLoss,
            optimizer=tf.keras.optimizers.Adam(lr=lr, decay=decay),
            metrics=['accuracy']
        )

    return model

def train(leagues, lstm, lossType, epochs, dropout, activation, units, lr, decay):

    # loss types:
        # 'categorical'
        # 'return'
        # 'goals'
        # 'goaldif'

    trainDf = loadData(leagues)

    date1 = datetime.strptime('2018-07-01', r'%Y-%m-%d')
    trainDf = trainDf.loc[trainDf['Date'] <= date1]
    trainDf = trainDf.loc[(trainDf['T_GamesPlayed_H'] >= 3) & (trainDf['T_GamesPlayed_A'] >= 3)]
    trainDf['L6H_Goals_HA'] = trainDf['L6H_Goals_HA'].apply(lambda x: np.asarray(ast.literal_eval(x)))

    if lossType=='return':
        tempLabelsH = trainDf[['HomeOdds','FTR']].apply(lambda x: (x.HomeOdds-0.05)*-1 if x.FTR == 'H' else 0.05, axis=1)
        tempLabelsD = trainDf[['DrawOdds','FTR']].apply(lambda x: (x.DrawOdds-0.05)*-1 if x.FTR == 'D' else 0.05, axis=1)
        tempLabelsA = trainDf[['AwayOdds','FTR']].apply(lambda x: (x.AwayOdds-0.05)*-1 if x.FTR == 'A' else 0.05, axis=1)
        y_train = pd.concat([tempLabelsH, tempLabelsD, tempLabelsA], axis=1, ignore_index=True).to_numpy()

    elif lossType=='goals':
        y_train = trainDf[['GoalsFor_H','GoalsFor_A']].to_numpy()

    elif lossType=='goaldif':
        y_train = trainDf[['GoalsFor_H','GoalsFor_A']].apply(lambda x: x.GoalsFor_H - x.GoalsFor_A, axis=1).to_numpy()

    else:
        y_train = trainDf['FTR'].apply(lambda x: 2 if x == 'H' else (1 if x == 'D' else 0)).to_numpy()

    if lstm==True:
        x_train1 = pd.concat([trainDf.iloc[:,15:18], trainDf.iloc[:,22:26], trainDf.iloc[:,34:36]], axis=1, ignore_index=True).to_numpy()
        x_train2 = np.asarray([i for i in trainDf.iloc[:,-1]])
        x_train = [x_train1, x_train2]

        model = buildModel(inputs1=x_train1, inputs2=x_train2, lstm=lstm, lossType=lossType, dropout=dropout, lr=lr, decay=decay)
        print(model.summary())

    else:
        x_train = pd.concat([trainDf.iloc[:,15:18], trainDf.iloc[:,22:26], trainDf.iloc[:,34:36]], axis=1, ignore_index=True).to_numpy()

        model = buildModel(inputs1=x_train, lstm=lstm, lossType=lossType, dropout=dropout, lr=lr, decay=decay)
        print(model.summary())

    model.fit(
        x_train, 
        y_train,
        batch_size=64,
        epochs=epochs,
        validation_split=0.2,
        callbacks = [tensorboard]
    )

    model.save('savedModels/' + modelName + '.h5')


train(
    leagues=leagues, 
    lstm=True, 
    lossType='categorical',
    epochs=100, 
    dropout=False, 
    activation='relu', 
    units=32,
    lr=1e-3, 
    decay=1e-5
)
