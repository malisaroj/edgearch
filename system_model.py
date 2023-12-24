import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU


def create_bilstm_model(units):
    model = Sequential()
    #First layer of BILSTM
    model.add(Bidirectional(LSTM(units=units, return_sequences = True), 
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    #Second layer of BILSTM
    model.add(Bidirectional(LSTM(units=units)))
    model.add(GRU(units= units, return_sequences=True))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')

    return model