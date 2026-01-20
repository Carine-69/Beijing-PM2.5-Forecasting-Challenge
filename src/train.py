import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import preprocess_train, preprocess_test

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# hyperperameters
#mumber of past hours used to predict the next hour
lookback = 24

#number of times the model sees full dataset
epochs = 30

#number of samples to be processed once
batch_size = 32

#learning rate

learning_rate = 0.01

#load training data

x_train, y_train, scaler_x, scaler_y = preprocess_train("../data/processed/train.csv", lookback=lookback)

n_timesteps = x_train.shape[1]
n_feature = x_train.shape[2] 

#build RNN model

rnn_model = Sequential([SimpleRNN(64, activation = "tanh", input_shape = (n_timesteps, n_feature)),
 Dense(1)
])

rnn_model.compile(
    optimizer = Adam(learning_rate=learning_rate),
    loss = "mse"
)

#build sltm model
lstm_model = Sequential([LSTM(64, activation= "tanh", input_shape = (n_timesteps, n_feature)),
Dense(1)])

lstm_model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss = "mse"
)

#train RNN

early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

print("\nTraining RNN...")
history_rnn = rnn_model.fit(
    x_train, y_train,
    epochs= epochs,
    batch_size=batch_size,
    callbacks= [early_stop],
    verbose=1
)


#train lstm
print("\nTraining LSTM...")
history_lstm = lstm_model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)

#load test data
x_test, test_index= preprocess_test(
    "../data/processed/test.csv",
    lookback=lookback,
    scaler_x= scaler_x
)

#make prediction
y_pred_rnn_scaled  = rnn_model.predict(x_test)
y_pred_lstm_scaled = lstm_model.predict(x_test)

#inverse scale back to real pm2.5
y_pred_rnn  = scaler_y.inverse_transform(y_pred_rnn_scaled)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled)


# save tets prediction csv
#creat dataframe
df_pred = pd.DataFrame({
    "datetime": test_index,
    "pm2.5_RNN":y_pred_rnn.flatten(),
    "pm2.5_LSTM": y_pred_lstm.flatten()
}
)

df_pred.to_csv("../data/processed/test_prediction.csv", index=False)

#plot predictions 

plt.figure(figsize= (12,5))
plt.plot(test_index, y_pred_rnn, label = "RNN prediction")
plt.plot(test_index, y_pred_lstm, label = "LSTM prediction")
plt.xlabel("Time")
plt.ylabel("pm2.5")
plt.title("RNN vs LSTM Predictions on Test Data")
plt.legend()
plt.grid(True)
plt.show()
