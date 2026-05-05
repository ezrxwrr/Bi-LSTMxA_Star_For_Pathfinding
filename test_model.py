import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

model_path = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\model_prediksi_rintangan.pth'
dataset_path = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\Dataset\annotations\bookstore\video0\annotations.txt'
scaler_path = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\scaler.pkl'
output_dir = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel'

if not os.path.exists(model_path) or not os.path.exists(dataset_path) or not os.path.exists(scaler_path):
    print("ERROR: File tidak ditemukan!")
    exit()

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=2, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        _, (hn, _) = self.lstm2(x)
        x = hn[-1] 
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

scaler = joblib.load(scaler_path)

df = pd.read_csv(dataset_path, sep=' ', header=None, 
                 names=['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'])

df['x'] = (df['xmin'] + df['xmax']) / 2
df['y'] = (df['ymin'] + df['ymax']) / 2

target_id = 0
df_single = df[df['track_id'] == target_id].sort_values(by='frame')

data_scaled = scaler.transform(df_single[['x', 'y']].values)

SEQ_LENGTH = 10
X_test, y_true = [], []
for i in range(len(data_scaled) - SEQ_LENGTH):
    X_test.append(data_scaled[i:i + SEQ_LENGTH])
    y_true.append(data_scaled[i + SEQ_LENGTH])

X_test_t = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
y_true_scaled = np.array(y_true)

with torch.no_grad():
    predictions_scaled = model(X_test_t).cpu().numpy()

predictions_orig = scaler.inverse_transform(predictions_scaled)
actual_orig = scaler.inverse_transform(y_true_scaled)

mse = mean_squared_error(y_true_scaled, predictions_scaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_scaled, predictions_scaled)

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
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.6f}', va='bottom')
plt.title('Error Metrics Evaluation')
plt.ylabel('Value')
plt.savefig(os.path.join(output_dir, 'metrik_error.png'))
plt.show()

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")