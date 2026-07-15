import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from model_def import ResNetSpeakerCNN # Requires model_def.py to be present

# Load speaker mapping
speaker_names = ['Speaker0026', 'Speaker0027', 'Speaker0028', 'Speaker0029', 'Speaker0030', 'Speaker0031', 'Speaker0032', 'Speaker0033', 'Speaker0034', 'Speaker0035', 'Speaker0036', 'Speaker0037', 'Speaker0038', 'Speaker0039', 'Speaker0040', 'Speaker0041', 'Speaker0042', 'Speaker0043', 'Speaker0044', 'Speaker0045', 'Speaker0046', 'Speaker0047', 'Speaker0048', 'Speaker0049', 'Speaker0050', 'Speaker_0000', 'Speaker_0001', 'Speaker_0002', 'Speaker_0003', 'Speaker_0004', 'Speaker_0005', 'Speaker_0006', 'Speaker_0007', 'Speaker_0008', 'Speaker_0009', 'Speaker_0010', 'Speaker_0011', 'Speaker_0012', 'Speaker_0013', 'Speaker_0014', 'Speaker_0015', 'Speaker_0016', 'Speaker_0017', 'Speaker_0018', 'Speaker_0019', 'Speaker_0020', 'Speaker_0021', 'Speaker_0023', 'Speaker_0024', 'Speaker_0025']

@st.cache_resource
def load_model():
    model = ResNetSpeakerCNN(num_classes=50)
    model.load_state_dict(torch.load('FINAL_speaker_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def extract_features(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000, duration=3.0)
    if len(audio) < 48000: audio = np.pad(audio, (0, 48000 - len(audio)))
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=512, hop_length=375)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)[:, :129]
    return torch.Tensor(log_mel_spec).unsqueeze(0).unsqueeze(0)

st.title("🎤 AI Speaker Recognition")
st.write("Identify the speaker from a 3-second audio clip.")

model = load_model()
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    with st.spinner('Analyzing voice...'):
        features = extract_features(uploaded_file)
        with torch.no_grad():
            outputs = model(features)
            _, pred = torch.max(outputs, 1)
            conf = F.softmax(outputs, dim=1)[0][pred].item()
        
        st.success(f"### Identified: {speaker_names[pred.item()]}")
        st.info(f"Confidence Level: {conf:.2%}")