import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

def formatX(X,wlen):
   return np.reshape(X, (-1, wlen, 1))

def formatY(Y,num_classes):
   Y = np.reshape(Y, (-1, 1,))
   return keras.utils.to_categorical(Y, num_classes)

import barcode.CNNWavenet as cnnwavenet
def trainCNN(indata,outpath):

    df = pd.read_parquet(indata)
    # train_df, test_df = train_test_split(df, test_size=4800)
    #
    # X_train  = np.concatenate(train_df['bc_signal'].to_numpy())  / 800
    # Y_train = train_df['bc'].to_numpy()
    # X_test = np.concatenate(test_df['bc_signal'].to_numpy())/ 800
    # Y_test = test_df['bc'].to_numpy()
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for index,row in df.iterrows():

        bc = row[1]
        signal = row[2]
        if index%12 > 10:
            X_test.append(signal)
            Y_test.append(bc)
        else:
            X_train.append(signal)
            Y_train.append(bc)

    trnas = sorted(df['bc'].unique())
    # name to index
    Y_train = list(map(lambda trna: trnas.index(trna), Y_train))
    Y_test = list(map(lambda trna: trnas.index(trna), Y_test))
    num_classes = np.unique(Y_train).size
    wlen = 4096+1024
    train_x = formatX(X_train, wlen)
    train_y = formatY(Y_train, num_classes)
    test_x = formatX(X_test, wlen)
    test_y = formatY(Y_test, num_classes)
    batch_size = 256
    epoch = 300
    lr = 0.0008
    modelCheckpoint = ModelCheckpoint(filepath= outpath,
                                      monitor='val_accuracy',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode='max',
                                      period=1)

    model = cnnwavenet.build_network(shape=(None, wlen, 1), num_classes=num_classes)
    optim = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    history = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, verbose=1,
                        shuffle=True, validation_data=(test_x, test_y),callbacks=[modelCheckpoint])

indata = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/out.pq"
outpath = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/weight"

trainCNN(indata,outpath)