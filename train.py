import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and basic cleaning
dataset_path = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\Dataset\annotations\bookstore\video0\annotations.txt'
df = pd.read_csv(dataset_path, sep=' ', header=None, 
                 names=['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'])

df['x'] = (df['xmin'] + df['xmax']) / 2
df['y'] = (df['ymin'] + df['ymax']) / 2
df = df[['frame', 'track_id', 'x', 'y']].sort_values(by=['track_id', 'frame'])

# Back to basics: Only x, y
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_raw = df[['x', 'y']].values 
y_raw = df[['x', 'y']].values 

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

joblib.dump(scaler_X, r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\scaler_X.pkl')
joblib.dump(scaler_y, r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\scaler_y.pkl')

def create_sequences(X_data, y_data, seq_length):
    X, y = [], []
    for i in range(len(X_data) - seq_length):
        X.append(X_data[i:i + seq_length])
        y.append(y_data[i + seq_length])
    return np.array(X), np.array(y)

# Small sequence to avoid memorizing long paths
SEQ_LENGTH = 8 
X, y = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

# Super lightweight Bi-LSTM
class SimpleBiLSTM(nn.Module):
    def __init__(self):
        super(SimpleBiLSTM, self).__init__()
        # Only 1 layer, 32 hidden units
        self.lstm = nn.LSTM(input_size=2, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32 * 2, 2)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        # Concatenate forward and backward hidden states
        x = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        x = self.fc(x)
        return x

model = SimpleBiLSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

scaler_amp = torch.amp.GradScaler('cuda')
epochs = 100
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 10 # Stop quickly if overfitting starts

model_name = r'C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\model_prediksi_rintangan_pro.pth'

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X_v, batch_y_v in test_loader:
            batch_X_v, batch_y_v = batch_X_v.to(device), batch_y_v.to(device)
            with torch.amp.autocast('cuda'):
                v_out = model(batch_X_v)
                v_loss = criterion(v_out, batch_y_v)
            val_loss += v_loss.item() * batch_X_v.size(0)
    
    val_loss /= len(test_loader.dataset)
    scheduler.step(val_loss)
    torch.cuda.empty_cache()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_name)
        patience_counter = 0
    else:
        patience_counter += 1
        
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03}/{epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

    if patience_counter >= early_stop_patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# Evaluation
model.load_state_dict(torch.load(model_name))
model.eval()
val_outputs_list = []
with torch.no_grad():
    for b_X, _ in test_loader:
        b_X = b_X.to(device)
        with torch.amp.autocast('cuda'):
            out = model(b_X)
        val_outputs_list.append(out.cpu().numpy())

val_outputs_np = np.concatenate(val_outputs_list, axis=0)
y_test_px = scaler_y.inverse_transform(y_test_t.numpy())
val_outputs_px = scaler_y.inverse_transform(val_outputs_np)

ade_px = np.mean(np.linalg.norm(y_test_px - val_outputs_px, axis=1))
fde_px = np.mean(np.linalg.norm(y_test_px - val_outputs_px, axis=1))
print(f"\nADE: {ade_px:.2f} px")
print(f"FDE: {fde_px:.2f} px")