import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, \
    root_mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, LSTM, Conv1D, MaxPooling1D, BatchNormalization, Bidirectional, Flatten, \
    Reshape
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras.optimizers import Nadam
import math
from keras.saving import register_keras_serializable


@register_keras_serializable(package="Custom")
def rmse(s_data, d_data):
    return root_mean_squared_error(s_data, d_data)


def Mae_error(user_data, pred_data):
    error = mean_absolute_percentage_error(user_data, pred_data)
    mae = mean_absolute_error(user_data, pred_data)
    return mae


def load_data(ndim):
    data = np.load("data.npy")
    target = np.load("target.npy")

    return data, target


def load_test_data(ndim):
    data = np.load("test_data.npy")
    target = np.load("test_target.npy")

    return data, target


def decoder(ndim):
    decoder_model = pickle.load(open("decoder_model_16500.sav", 'rb'))
    return decoder_model


def ks_distance_loss(y_true, y_pred):
    y_true_cdf = np.cumsum(y_true / np.sum(y_true))
    y_pred_cdf = np.cumsum(y_pred / np.sum(y_pred))
    ks_distance = np.max(np.abs(y_true_cdf - y_pred_cdf))

    return ks_distance


def frechet_distance(y_true, y_pred):
    y_true_cumsum = tf.cumsum(y_true, axis=-1)
    y_pred_cumsum = tf.cumsum(y_pred, axis=-1)
    print("cumsum:", y_pred_cumsum)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true_cumsum - y_pred_cumsum), axis=-1)))


def ks_distance(y_true, y_pred):
    y_true_cdf = tf.cumsum(y_true / tf.reduce_sum(y_true))
    y_pred_cdf = tf.cumsum(y_pred / tf.reduce_sum(y_pred))
    return tf.reduce_max(tf.abs(y_true_cdf - y_pred_cdf))


def LSTM_model(x_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0, input_shape=(x_train.shape[1], x_train.shape[2]), name="masking"),
        Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2)),
        Dropout(0.2),

        LSTM(64, return_sequences=False),
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        Dense(8, activation='sigmoid', name="latent_space")  # 8D latent space output
    ])

    model.compile(optimizer=Nadam(learning_rate=0.001), loss='mse',
                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)

    # Train the model
    history = model.fit(x_train, y_train, epochs=2000, batch_size=32, validation_split=0.2, verbose=1
                        , callbacks=[early_stopping, reduce_lr])

    return model, history


ndim = 4
y_train_enc = []
y_test_enc = []

train_truth = []
test_truth = []

data, target = load_data(ndim)
x_test, y_test = load_test_data(ndim)
x_train, y_train = data, target
for i in range(y_train.shape[0]):
    y_train_enc.append(y_train[i][:8])  # Extracting the encoded data TRAIN
    train_truth.append(y_train[i][8:])  # Extracting the encoded data

for i in range(y_test.shape[0]):
    y_test_enc.append(y_test[i][:8])   # Extracting the encoded data TEST
    test_truth.append(y_test[i][8:])   # Extracting the encoded data

y_train_enc = np.array(y_train_enc)
y_test_enc = np.array(y_test_enc)

train_truth = np.array(train_truth)
test_truth = np.array(test_truth)

# Training code
model, history = LSTM_model(x_train, y_train_enc)

# Print the summary of the model
model.summary()

# save the model to disk
filename = "LSTM_model.sav"
pickle.dump(model, open(filename, 'wb'))

# Evaluate the model on the test data

test_loss, test_mae, test_rmse = model.evaluate(x_test, y_test_enc)
print(f'Test MAE: {test_loss}, Test MSE: {test_mae}, Test RMSE : {test_rmse}')

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.grid(True)
pyplot.title("MSE Loss")
pyplot.show()

plt.plot(history.history['mae'], markersize=5)
plt.plot(history.history['val_mae'], markersize=5)
plt.title('MAE Loss')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
pyplot.grid(True)
plt.show()

plt.plot(history.history['root_mean_squared_error'], markersize=5)
plt.plot(history.history['val_root_mean_squared_error'], markersize=5)
plt.title('RMSE Loss')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
pyplot.grid(True)
plt.show()


# Evaluate the model

model = pickle.load(open("AE_" + str(ndim) + "/weighted/models/LSTM_model.sav", 'rb'))

model.summary()
y_pred = model.predict(x_test)

decoded_values_pred_list = []
decoded_values_test_list = []
non_eq_mae = 0
eq_mae = 0
non_eq_ks = 0
eq_ks = 0

decoder_model = decoder(ndim)
decoded_values_pre = decoder_model.predict(y_pred)
decoded_values_tes = decoder_model.predict(y_test_enc)

np.save("LSTM_test_data.npy", test_truth)
np.save("LSTM_decoded_pred.npy", decoded_values_pre)

for i in range(y_pred.shape[0]):
    non_eq_mae = non_eq_mae + Mae_error(test_truth[i][:29], decoded_values_pre[i][:29])
    non_eq_ks = non_eq_ks + ks_distance_loss(test_truth[i][:29], decoded_values_pre[i][:29])

    eq_mae = eq_mae + Mae_error(test_truth[i][29:], decoded_values_pre[i][29:])
    eq_ks = eq_mae + ks_distance_loss(test_truth[i][29:], decoded_values_pre[i][29:])

print("Non: Eq _cl MAE Mean:", non_eq_mae / y_pred.shape[0])
print("Non: Eq _cl KS value Mean:", non_eq_ks / y_pred.shape[0])
print("Eq _cl MAE Mean:", eq_mae / y_pred.shape[0])
print("Eq _cl KS Value Mean:", eq_ks / y_pred.shape[0])

rand_range = np.random.randint(0, 200, 6)

for i in rand_range:
    plt.figure(figsize=(12, 6))

    plt.plot(range(len(test_truth[i][:29])), test_truth[i][:29], c='g', linestyle='--', linewidth=0.7,
             label='Non-Equilibrium : Ground Truth ')

    plt.plot(range(len(decoded_values_pre[i][:29])), decoded_values_pre[i][:29], c='r',
             label='Non-Equilibrium : Predicted', linewidth=0.7)

    plt.plot(range(len(test_truth[i][29:])), test_truth[i][29:], c='b', linestyle='--', linewidth=0.7,
             label='Equilibrium : Ground Truth ')

    plt.plot(range(len(decoded_values_pre[i][29:])), decoded_values_pre[i][29:], c='y',
             label='Equilibrium : Predicted', linewidth=0.7)
    mae = Mae_error(decoded_values_pre[i][:29], test_truth[i][:29])
    rmse_ = rmse(decoded_values_pre[i][:29], test_truth[i][:29])
    ks_dist = ks_distance_loss(decoded_values_pre[i][:29], test_truth[i][:29])

    plt.legend(fontsize=15, loc='upper right')
    plt.show()
plt.clf()

plt.figure(figsize=(12, 6))
for i in range(y_pred.shape[0]):
    plt.plot(range(len(decoded_values_tes[i][:29])), decoded_values_tes[i][:29], c='y', linewidth=0.7)
plt.title("Test values - Decoded by AE (Non Equilibrium)", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(loc='upper right')
plt.show()
plt.clf()

# Plotting all the test distributions

plt.figure(figsize=(12, 6))

for i in range(y_pred.shape[0]):
    plt.plot(range(len(test_truth[i][:29])), test_truth[i][:29], c='g', linewidth=0.7)
plt.title("Truth Value (Non Equilibrium)", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(loc='upper right')
plt.show()
plt.clf()

plt.figure(figsize=(12, 6))

for i in range(y_pred.shape[0]):
    plt.plot(range(len(decoded_values_pre[i][:29])), decoded_values_pre[i][:29], c='r', linewidth=0.7)
    plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.title("Predicted by LSTM - Decoded by AE( Non-Equilibrium)", fontsize=14, fontweight='bold')
plt.show()
plt.clf()

plt.figure(figsize=(12, 6))
for i in range(y_pred.shape[0]):
    plt.plot(range(len(decoded_values_pre[i][29:])), decoded_values_pre[i][29:], c='r', linewidth=0.7)
    plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.title("Predicted by LSTM - Decoded by AE (Equilibrium)", fontsize=14, fontweight='bold')
plt.show()
plt.clf()

plt.figure(figsize=(12, 6))

for i in range(y_pred.shape[0]):
    plt.plot(range(len(decoded_values_tes[i][29:])), decoded_values_tes[i][29:], c='y', linewidth=0.7)
    plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.title("Test values - Decoded by AE ( Equilibrium)", fontsize=14, fontweight='bold')
plt.show()
plt.clf()

plt.figure(figsize=(12, 6))
for i in range(y_pred.shape[0]):
    plt.plot(range(len(test_truth[i][29:])), test_truth[i][29:], c='g', linewidth=0.7)
    plt.legend(loc='upper right')
plt.title("Truth Value ( Equilibrium)", fontsize=14, fontweight='bold')
plt.show()
plt.clf()
