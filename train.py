import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

dataset_path = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\Dataset\annotations\bookstore\video0\annotations.txt'

df = pd.read_csv(dataset_path, sep=' ', header=None, 
                 names=['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'])

df['x'] = (df['xmin'] + df['xmax']) / 2
df['y'] = (df['ymin'] + df['ymax']) / 2
df = df[['frame', 'track_id', 'x', 'y']].sort_values(by=['track_id', 'frame'])

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['x', 'y']].values)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 10 
X, y = create_sequences(data_scaled, SEQ_LENGTH)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(128, input_shape=(SEQ_LENGTH, 2), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse')

print("Memulai proses pelatihan model (Urutan Tetap)...")
# shuffle=False sangat penting untuk data time-series!
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), shuffle=False)

model_name = 'model_prediksi_rintangan.h5'
model.save(model_name)
print(f"\nSelesai! Model disimpan sebagai: {model_name}")