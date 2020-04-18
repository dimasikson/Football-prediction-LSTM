
import tensorflow as tf

import os
import numpy as np
import pandas as pd
from datetime import datetime
from utils import loadData
import time
import ast


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

def predict(leagues, lstm=True, lossType='categorical', modelNum=-1):

    fnames = []

    for filename in os.listdir('savedModels'):
        if filename.startswith(leagues[0]):
            fnames.append(filename)

    modelFname = fnames[modelNum]

    model = tf.keras.models.load_model(os.path.join('savedModels',modelFname))

    testDf = loadData(leagues)

    date1 = datetime.strptime('2018-07-01', r'%Y-%m-%d')
    testDf = testDf.loc[testDf['Date'] > date1]
    testDf = testDf.loc[(testDf['T_GamesPlayed_H'] >= 3) & (testDf['T_GamesPlayed_A'] >= 3)]
    testDf['L6H_Goals_HA'] = testDf['L6H_Goals_HA'].apply(lambda x: np.asarray(ast.literal_eval(x), dtype=np.float32))

    if lstm==True:
        x_test1 = pd.concat([testDf.iloc[:,15:18], testDf.iloc[:,22:26], testDf.iloc[:,34:36]], axis=1, ignore_index=True).to_numpy()
        x_test2 = np.asarray([i for i in testDf.iloc[:,-1]])
        output = model.predict([x_test1, x_test2])

    else:
        x_test = pd.concat([testDf.iloc[:,15:18], testDf.iloc[:,22:26], testDf.iloc[:,34:36]], axis=1, ignore_index=True).to_numpy()        
        output = model.predict(x_test)

    print(output)

    if lossType=='goals':
        testDf['HPredict_Goals'] = output[:,0]
        testDf['APredict_Goals'] = output[:,1]
    if lossType=='goaldif':
        testDf['Predict_Goaldif'] = output[:,0]
    else:
        testDf['HPredict'] = output[:,2]
        testDf['DPredict'] = output[:,1]
        testDf['APredict'] = output[:,0]

    testDf.to_csv('predictedOutput/' + modelFname + '.csv')

predict(leagues=leagues, lstm=True, lossType='categorical', modelNum=-1)
