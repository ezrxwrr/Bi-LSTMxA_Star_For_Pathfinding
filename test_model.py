import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

model_path = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\model_prediksi_rintangan.h5'
dataset_path = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\Dataset\annotations\bookstore\video0\annotations.txt'
output_dir = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel'

if not os.path.exists(model_path) or not os.path.exists(dataset_path):
    print("ERROR: File tidak ditemukan!")
    exit()

model = load_model(model_path, compile=False)

df = pd.read_csv(dataset_path, sep=' ', header=None, 
                 names=['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'])

df['x'] = (df['xmin'] + df['xmax']) / 2
df['y'] = (df['ymin'] + df['ymax']) / 2

target_id = 0
df_single = df[df['track_id'] == target_id].sort_values(by='frame')

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_single[['x', 'y']].values)

SEQ_LENGTH = 10
X_test, y_true = [], []
for i in range(len(data_scaled) - SEQ_LENGTH):
    X_test.append(data_scaled[i:i + SEQ_LENGTH])
    y_true.append(data_scaled[i + SEQ_LENGTH])

X_test = np.array(X_test)
y_true = np.array(y_true)

predictions = model.predict(X_test)
predictions_orig = scaler.inverse_transform(predictions)
actual_orig = scaler.inverse_transform(y_true)

mse = mean_squared_error(actual_orig, predictions_orig)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_orig, predictions_orig)

plt.figure(figsize=(10, 6))
plt.plot(actual_orig[-50:, 0], actual_orig[-50:, 1], 'g-', linewidth=2, label='Actual')
plt.plot(predictions_orig[-50:, 0], predictions_orig[-50:, 1], 'r--', linewidth=2, label='Prediction')
plt.scatter(actual_orig[-1, 0], actual_orig[-1, 1], color='green', s=100)
plt.scatter(predictions_orig[-1, 0], predictions_orig[-1, 1], color='red', marker='X', s=100)
plt.title(f'Trajectory Analysis (ID: {target_id})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'hasil_kordinat.png'))
plt.show()

label_counts = df.groupby('track_id')['label'].first().value_counts()
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Object Distribution')
plt.savefig(os.path.join(output_dir, 'distribusi_objek.png'))
plt.show()

metrics = ['MSE', 'RMSE', 'MAE']
values = [mse, rmse, mae]
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['blue', 'orange', 'green'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 6), va='bottom')
plt.title('Error Metrics Evaluation')
plt.ylabel('Value')
plt.savefig(os.path.join(output_dir, 'metrik_error.png'))
plt.show()

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")