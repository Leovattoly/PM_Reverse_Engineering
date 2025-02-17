import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split


# Normalization Constant
NORMALIZE_VALUE = 1641775


# Load and Normalize Data
def load_data(file_path):
    data = pd.read_csv(file_path)
    non_eq_cl_list, eq_cl_list = [], []

    for i in range(data.shape[0]):
        non_eq_cl = [data.loc[i, f'non_eq_cl{kk}'] / NORMALIZE_VALUE for kk in range(29)]
        eq_cl = [data.loc[i, f'eq_cl{kk}'] / NORMALIZE_VALUE for kk in range(29)]

        non_eq_cl_list.append(non_eq_cl)
        eq_cl_list.append(eq_cl)

    non_eq_cl_arr = np.array(non_eq_cl_list)
    eq_cl_arr = np.array(eq_cl_list)

    return np.concatenate((non_eq_cl_arr, eq_cl_arr), axis=1)


# Load datasets
train_data = load_data("Data/train.csv")
test_data = load_data("Data/test_test_dataset.csv")


# Train-test split
train_data, val_data = train_test_split(train_data, test_size=0.20, random_state=42)


# Model Parameters
input_dim = train_data.shape[1]
encoding_dim = 8  # Latent space size

# Define Encoder
input_layer = Input(shape=(input_dim,))
encoder_hidden1 = (Dense(64, activation='relu')(input_layer))
encoder_hidden1 = Dropout(0.2)(encoder_hidden1)
encoder_hidden2 = (Dense(32, activation='relu')(encoder_hidden1))
encoder_hidden2 = BatchNormalization()(encoder_hidden2)
encoder_hidden3 = (Dense(16, activation='relu')(encoder_hidden2))
encoder_hidden3 = BatchNormalization()(encoder_hidden3)
encoder_hidden4 = (Dense(8, activation='relu')(encoder_hidden3))

encoded = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(1e-6))(encoder_hidden4)

# Decoder
decoder_hidden1 = (Dense(8, activation='relu')(encoded))
decoder_hidden1 = BatchNormalization()(decoder_hidden1)
decoder_hidden2 = Dense(16, activation='relu')(decoder_hidden1)
decoder_hidden2 = BatchNormalization()(decoder_hidden2)
decoder_hidden3 = Dense(32, activation='relu')(decoder_hidden2)
decoder_hidden3 = BatchNormalization()(decoder_hidden3)
decoder_hidden4 = (Dense(64, activation='relu')(decoder_hidden3))
decoded = Dense(input_dim, activation='sigmoid')(decoder_hidden4)

# Build Autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Nadam(learning_rate=0.001), loss='mse',
                    metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae'])

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50)
early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

# Train Autoencoder
history = autoencoder.fit(train_data, train_data,
                          epochs=5000, batch_size=64, shuffle=True,
                          validation_data=(val_data, val_data),
                          callbacks=[early_stopping, lr_scheduler])

# Extract Encoder and Decoder
encoder = Model(input_layer, encoded)
X_encoded = encoder.predict(test_data)

decoder_input = Input(shape=(encoding_dim,))
decoder_layer_1 = autoencoder.layers[-8]  # Dense(8)
decoder_layer_2 = autoencoder.layers[-7]  # BatchNormalization()
decoder_layer_3 = autoencoder.layers[-6]  # Dense(16)
decoder_layer_4 = autoencoder.layers[-5]  # BatchNormalization()
decoder_layer_5 = autoencoder.layers[-4]  # Dense(32)
decoder_layer_6 = autoencoder.layers[-3]  # BatchNormalization()
decoder_layer_7 = autoencoder.layers[-2]  # Dense(64)
decoder_layer_8 = autoencoder.layers[-1]  # Dense(input_dim)

decoder = Model(decoder_input,
                decoder_layer_8(decoder_layer_7(decoder_layer_6(
                    decoder_layer_5(
                        decoder_layer_4(decoder_layer_3(decoder_layer_2(decoder_layer_1(decoder_input)))))))))
X_reconstructed = decoder.predict(X_encoded)

# Evaluate Model
test_mse, test_mae, test_rmse,  = autoencoder.evaluate(test_data, test_data)

# Save Results
np.save("AE_test_data_"+str(16500)+".npy", test_data)
np.save("AE_Encoded_test_"+str(16500)+".npy", X_encoded)
np.save("AE_decoded_test_"+str(16500)+".npy", X_reconstructed)

pickle.dump(encoder, open('encoder_model_' + str(16500) + '.sav', 'wb'))
pickle.dump(decoder, open('decoder_model_' + str(16500) + '.sav', 'wb'))

# Print Results
print(f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test MSE: {test_mse:.5f}")


