import streamlit as st
from streamlit_wavesurfer import wavesurfer
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import os
from source_.model import AudioCNN
from source_.pipeline import extract_features


# 1. Page Configuration & UI Styling
st.set_page_config(
    page_title="Urban Sound Classifier", 
    page_icon="🔊", 
    layout="centered"
)

# --- BACKGROUND IMAGE INJECTION ---
def set_bg_image():
    # Use a high-quality, dark acoustic or abstract engineering wave background URL
    bg_img_url = "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=1920"
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.85)), url("{bg_img_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Make content boxes readable over the background */
        .stMarkdown, .stFileUploader, div[data-testid="stAudio"] {{
            background-color: rgba(20, 20, 20, 0.6);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Run the function to apply the styling
set_bg_image()

st.title("🔊 Real-Time Urban Sound Classifier")
st.write(
    "Upload a short environmental audio clip (.wav) to see the deep learning pipeline "
    "transform the raw audio into a Log-Mel Spectrogram and run real-time inference."
)

# 2. Cached Model Loading (Ensures fast inference by loading weights once)
@st.cache_resource
def load_deep_learning_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN()
    
    weights_path = 'audio_cnn_baseline.pth'
    if not os.path.exists(weights_path):
        st.error(f"Critical Error: Weights file '{weights_path}' missing from root directory.")
        return None, device
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # Freeze BatchNorm and Dropout behaviors for inference
    return model, device

model, device = load_deep_learning_model()

# 3. File Uploader Interface
uploaded_file = st.file_uploader("Choose an environmental sound file (.wav)...", type=["wav"])

if uploaded_file is not None:
    st.markdown("### 🎧 Play Audio")
    
    # Reset file pointer to read from the start
    uploaded_file.seek(0)
    audio_bytes = uploaded_file.read()
    
    # Render the interactive WhatsApp-style scrubbing waveform
    wavesurfer(audio_bytes, regions=[])         # Height matching your spectrogram matrix geom)
    uploaded_file.seek(0)
    
    # 4. Trigger Classification on Button Click
    if st.button("🚀 Analyze Acoustic Signature"):
        if model is None:
            st.error("Model is not loaded properly. Check your .pth weights file.")
        else:
            with st.spinner("Extracting time-frequency features via Librosa..."):
                
                # Execute the exact feature extraction logic used during training
                features = extract_features(uploaded_file)
                
                if features is not None:
                    # Shape raw matrix to match PyTorch 2D CNN input expectations:
                    # Shape: (Batch Size = 1, Channels = 1, Height = 128, Width = 173)
                    tensor_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # 5. Run Core Model Pass
                    with torch.no_grad():
                        logits = model(tensor_input)
                        probabilities = F.softmax(logits, dim=1)
                        confidence, class_idx = torch.max(probabilities, 1)
                    
                    # Target class taxonomy mapping
                    class_names = [
                        'Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark', 
                        'Drilling', 'Engine Idling', 'Gun Shot', 'Jackhammer', 'Siren', 'Street Music'
                    ]
                    
                    predicted_label = class_names[class_idx.item()]
                    confidence_percentage = confidence.item() * 100
                    
                    # 6. Render Output Visualizations
                    st.markdown("---")
                    st.subheader(f"Prediction: **{predicted_label}**")
                    
                    # Render a visual progress bar indicating confidence intensity
                    st.progress(float(confidence.item()))
                    st.write(f"Model Confidence Score: **{confidence_percentage:.2f}%**")
                    
                    # Contextual styling alerts based on safety categories
                    if predicted_label in ['Gun Shot', 'Siren', 'Car Horn']:
                        st.warning(f"🚨 Hazard/Anomaly Detected: High likelihood of {predicted_label.lower()} event.")
                    else:
                        st.success(f"✅ Normal Urban Acoustic Signature Verified.")
                else:
                    st.error("Feature engineering pipeline failed. Verify that the file is an uncorrupted PCM WAV track.")
