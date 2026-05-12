import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# 1. Configuration
model_path = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\model_prediksi_rintangan_pro.pth'
dataset_path = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\Dataset\annotations\bookstore\video0\annotations.txt'
scaler_path = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\scaler_y.pkl'
output_dir = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Model Architecture (Must match your latest 2.14px model)
class SimpleBiLSTM(nn.Module):
    def __init__(self):
        super(SimpleBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32 * 2, 2)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        x = self.fc(x)
        return x

# 3. Load Model and Scaler
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    exit()

model = SimpleBiLSTM().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

scaler = joblib.load(scaler_path)

# 4. Data Preparation
df = pd.read_csv(dataset_path, sep=' ', header=None, 
                 names=['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'])

df['x'] = (df['xmin'] + df['xmax']) / 2
df['y'] = (df['ymin'] + df['ymax']) / 2

# Scale full dataset for global metrics
data_scaled = scaler.transform(df[['x', 'y']].values)

SEQ_LENGTH = 8 
X, y_true = [], []
for i in range(len(data_scaled) - SEQ_LENGTH):
    X.append(data_scaled[i:i + SEQ_LENGTH])
    y_true.append(data_scaled[i + SEQ_LENGTH])

X_test_t = torch.tensor(np.array(X), dtype=torch.float32).to(device)

# 5. Global Inference
print("Calculating global metrics...")
with torch.no_grad():
    with torch.amp.autocast('cuda'):
        predictions_scaled = model(X_test_t).cpu().numpy()

predictions_px = scaler.inverse_transform(predictions_scaled)
actual_px = scaler.inverse_transform(np.array(y_true))

# 6. Global Metrics Calculation
distances = np.linalg.norm(actual_px - predictions_px, axis=1)
ade = np.mean(distances)
fde = ade  # Based on 1-step prediction
mse = mean_squared_error(actual_px, predictions_px)
rmse = np.sqrt(mse)

print("\n" + "="*35)
print("      FINAL RESEARCH METRICS")
print("="*35)
print(f"ADE  : {ade:.4f} px")
print(f"FDE  : {fde:.4f} px")
print(f"MSE  : {mse:.4f} px^2")
print(f"RMSE : {rmse:.4f} px")
print("="*35)

# 7. Targeted Visualization (Finding a moving track)
# We pick a track_id that actually moves to avoid the "blank/static dot" issue
track_movements = df.groupby('track_id').apply(lambda x: np.linalg.norm(x[['x', 'y']].iloc[0] - x[['x', 'y']].iloc[-1]))
target_id = track_movements.idxmax() # Picks the track with the longest distance traveled
print(f"Visualizing Track ID: {target_id} (Longest Movement)")

df_target = df[df['track_id'] == target_id].sort_values(by='frame')
target_scaled = scaler.transform(df_target[['x', 'y']].values)

X_tar, y_tar = [], []
for i in range(len(target_scaled) - SEQ_LENGTH):
    X_tar.append(target_scaled[i:i + SEQ_LENGTH])
    y_tar.append(target_scaled[i + SEQ_LENGTH])

X_tar_t = torch.tensor(np.array(X_tar), dtype=torch.float32).to(device)
with torch.no_grad():
    pred_tar_scaled = model(X_tar_t).cpu().numpy()

pred_tar_px = scaler.inverse_transform(pred_tar_scaled)
actual_tar_px = scaler.inverse_transform(np.array(y_tar))

# 8. Plotting
plt.figure(figsize=(10, 6))
plt.plot(actual_tar_px[:, 0], actual_tar_px[:, 1], 'g-', label='Ground Truth', linewidth=2)
plt.plot(pred_tar_px[:, 0], pred_tar_px[:, 1], 'r--', label='LSTM Prediction', linewidth=1.5)
plt.title(f'Trajectory Prediction Accuracy (Track ID: {target_id})')
plt.xlabel('X Coordinate (Pixels)')
plt.ylabel('Y Coordinate (Pixels)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, 'hasil_kordinat_final.png'))
plt.show()

# 9. Metric Bar Chart
metrics_name = ['ADE', 'FDE', 'RMSE']
metrics_val = [ade, fde, rmse]
plt.figure(figsize=(8, 6))
bars = plt.bar(metrics_name, metrics_val, color=['#3498db', '#e74c3c', '#2ecc71'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
plt.title('Final Error Evaluation (Pixels)')
plt.ylabel('Value (Pixels)')
plt.savefig(os.path.join(output_dir, 'metrik_error_final.png'))
plt.show()