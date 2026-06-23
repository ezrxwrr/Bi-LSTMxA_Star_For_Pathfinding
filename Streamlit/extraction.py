import pandas as pd
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class TrajectoryLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

model = TrajectoryLSTM()
model_path = r"C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Streamlit\model_prediksi_rintangan_pro.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

txt_path = r"C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Artikel\Dataset\annotations\bookstore\video0\annotations.txt"
kolom = ['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
df = pd.read_csv(txt_path, sep=' ', names=kolom)

df = df[df['label'] == 'Pedestrian'].copy()
df['x'] = (df['xmin'] + df['xmax']) / 2
df['y'] = (df['ymin'] + df['ymax']) / 2
df = df.sort_values(by=['track_id', 'frame'])

evaluated_samples = []

for track_id, group in df.groupby('track_id'):
    if len(group) >= 20:
        coords = group[['x', 'y']].values.tolist()
        
        past_list = coords[:8]
        future_list = coords[8:20]
        
        # Lowered movement threshold to 10 pixels to catch more candidates
        start_pt = np.array(past_list[0])
        end_pt = np.array(future_list[-1])
        if np.linalg.norm(end_pt - start_pt) < 10:
            continue
            
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0, 0], [1920, 1080]]))
        
        past_scaled = scaler.transform(np.array(past_list))
        past_tensor = torch.as_tensor(past_scaled, dtype=torch.float32).unsqueeze(0)
        
        pred_list = []
        current_input = past_tensor
        
        with torch.no_grad():
            for _ in range(12):
                pred = model(current_input) 
                pred_list.append(pred.squeeze().numpy())
                pred_expanded = pred.unsqueeze(1)
                current_input = torch.cat((current_input[:, 1:, :], pred_expanded), dim=1)
        
        pred_np = scaler.inverse_transform(np.array(pred_list))
        future_np = np.array(future_list)
        
        gt_vector = future_np[-1] - future_np[0]
        pred_vector = pred_np[-1] - pred_np[0]
        
        norm_gt = np.linalg.norm(gt_vector)
        norm_pred = np.linalg.norm(pred_vector)
        
        if norm_gt > 0 and norm_pred > 0:
            cos_sim = np.dot(gt_vector, pred_vector) / (norm_gt * norm_pred)
        else:
            cos_sim = -1
            
        mse = np.mean((pred_np - future_np)**2)
        
        evaluated_samples.append({
            "id": f"Pedestrians {track_id}",
            "past": past_list,
            "future": future_list,
            "score": mse,
            "cos_sim": cos_sim
        })

# Sort primarily by highest Cosine Similarity (direction), then lowest MSE (distance)
evaluated_samples.sort(key=lambda x: (-x['cos_sim'], x['score']))
best_samples = evaluated_samples[:5]

samples_dict = {item["id"]: {"past": item["past"], "future": item["future"]} for item in best_samples}

output_path = r"C:\Users\Pavilion\Documents\Tugas\Semester 4\RTI\Streamlit\sample.json"
with open(output_path, 'w') as f:
    json.dump(samples_dict, f, indent=4)

print(f"Extraction complete. Found {len(best_samples)} best available samples.")