import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class TrajectoryLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

st.set_page_config(page_title="SDD Obstacle Prediction", layout="centered")
st.title("Pedestrian Trajectory Prediction")

@st.cache_resource
def load_model():
    model = TrajectoryLSTM()
    model_path = os.path.join(BASE_DIR, "model_prediksi_rintangan_pro.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "sample.json")
    if not os.path.exists(data_path):
        return None
    with open(data_path, 'r') as f:
        return json.load(f)

data_samples = load_data()

if data_samples is None:
    st.error("Sample Not Found.")
else:
    st.sidebar.header("Settings")
    pilihan = st.sidebar.selectbox("Choose Pedestrian Sample:", list(data_samples.keys()))
    
    if st.button("Execute"):
        past_list = data_samples[pilihan]["past"]
        future_list = data_samples[pilihan]["future"]
        
        past_np_raw = np.array(past_list)
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0, 0], [1920, 1080]]))
        
        past_scaled = scaler.transform(past_np_raw)
        past_tensor = torch.as_tensor(past_scaled, dtype=torch.float32).unsqueeze(0)
        
        pred_list = []
        current_input = past_tensor
        
        with torch.no_grad():
            for _ in range(12):
                pred = model(current_input) 
                pred_list.append(pred.squeeze().numpy())
                
                pred_expanded = pred.unsqueeze(1)
                current_input = torch.cat((current_input[:, 1:, :], pred_expanded), dim=1)
        
        past_np = past_np_raw
        future_np = np.array(future_list)
        pred_np_scaled = np.array(pred_list)
        
        pred_np = scaler.inverse_transform(pred_np_scaled)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(past_np[:, 0], past_np[:, 1], 'bo-', label='Input (Past 8 Frames)')
        ax.plot(future_np[:, 0], future_np[:, 1], 'go-', label='Ground Truth', alpha=0.6)
        ax.plot(pred_np[:, 0], pred_np[:, 1], 'ro--', label='LSTM Prediction')
        
        ax.scatter(past_np[0, 0], past_np[0, 1], color='black', marker='s', s=100, label='Start Point')
        
        ax.set_title(f"Trajectory Visualization: {pilihan}")
        ax.set_xlabel("X Position (Pixels)")
        ax.set_ylabel("Y Position (Pixels)")
        ax.legend()
        ax.grid(True)
        
        ax.set_aspect('equal', adjustable='datalim')
        
        center_x = past_np[-1, 0]
        center_y = past_np[-1, 1]
        window_size = 30
        
        ax.set_xlim(center_x - window_size, center_x + window_size)
        ax.set_ylim(center_y - window_size, center_y + window_size)

        mse = np.mean((pred_np - future_np)**2)
        
        st.pyplot(fig)
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")