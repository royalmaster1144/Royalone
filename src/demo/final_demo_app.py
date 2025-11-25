import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from pathlib import Path
import numpy as np
from src.utils.device import get_device
from src.inference.speaker_encoder import load_model as load_spk, infer as infer_spk
from src.inference.tacotron2_infer import load_model as load_taco, infer as infer_taco
from src.inference.vocoder_infer import load_model as load_voc, infer as infer_voc
from src.audio.audio_utils import load_wav, save_wav, normalize_audio, extract_mel
from src.text.text_cleaner import clean_text, text_to_sequence

st.set_page_config(page_title="Voice Clone", layout="wide",page_icon="./image.png")

st.title("Voice Clone")

# Sidebar: device / model options
with st.sidebar:
    device_choice = st.selectbox("Device", ["GPU", "CPU"])  
    half = st.checkbox("Use half precision (fp16) for speed", value=False)
    st.markdown("**Models**")
    spk_path = st.text_input("Speaker encoder path", "models/speaker_encoder.pt")
    taco_path = st.text_input("Tacotron2 path", "models/tacotron2.pt")
    voc_path = st.text_input("Vocoder path", "models/vocoder.pt")

# Load models (cached)
@st.cache_resource
def load_models(spk, taco, voc, device, half):
    device_obj = get_device(prefer_gpu=(device=="GPU"))
    spk_m = load_spk(spk, device_obj, half)
    taco_m = load_taco(taco, device_obj, half)
    voc_m = load_voc(voc, device_obj, half)
    return spk_m, taco_m, voc_m, device_obj

if st.button("Load models"):
    try:
        spk_m, taco_m, voc_m, device_obj = load_models(spk_path, taco_path, voc_path, device_choice, half)
        st.success("Models loaded.")
    except Exception as e:
        st.error(f"Failed to load models: {e}")

# Main UI
col1, col2 = st.columns([1,1])
with col1:
    st.header("Input")
    input_method = st.radio("Input Method", ["Upload File", "Record Voice"])
    
    uploaded_files = []
    if input_method == "Upload File":
        files = st.file_uploader("Upload reference voice (wav)", type=["wav","mp3"], accept_multiple_files=True)
        if files:
            uploaded_files = files
    else:
        if "audio_key" not in st.session_state:
            st.session_state.audio_key = 0
            
        audio_value = st.audio_input("Record your voice", key=f"audio_recorder_{st.session_state.audio_key}")
        if audio_value:
            uploaded_files = [audio_value]
            
        if st.button("Delete Recording"):
            st.session_state.audio_key += 1
            st.rerun()

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) ready for processing.**")
        for f in uploaded_files:
             st.download_button(f"Save Recording ({f.name})", data=f.getvalue(), file_name=f.name, mime="audio/wav", key=f"save_{f.name}")

    text = st.text_area("Text to synthesize", value="Hello, this is a demo.", height=200)
    st.markdown("**Emotion sliders**")
    happy = st.slider("Happy", 0.0, 1.0, 0.0)
    sad = st.slider("Sad", 0.0, 1.0, 0.0)
    angry = st.slider("Angry", 0.0, 1.0, 0.0)
    neutral = st.slider("Neutral", 0.0, 1.0, 1.0)
    generate = st.button("Generate")

with col2:
    st.header("Output")
    output_audio = None
    status = st.empty()
    if generate:
        if not uploaded_files:
            st.error("Please upload or record reference voice.")
        else:
            status.info("Processing inputs...")
            
            for i, uploaded in enumerate(uploaded_files):
                st.markdown(f"### Processing: {uploaded.name}")
                
                # save temp file
                tmp_path = Path(f"tmp/ref_{i}_{uploaded.name}")
                tmp_path.parent.mkdir(exist_ok=True)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                

                wav, sr = load_wav(str(tmp_path), sr=16000)
                wav = normalize_audio(wav)
                # status.info(f"Generating speaker embedding for {uploaded.name}...") # Avoid spamming status
                spk_m, taco_m, voc_m, device_obj = load_models(spk_path, taco_path, voc_path, device_choice, half)
                emb = infer_spk(spk_m, wav, sample_rate=sr)
                # status.info(f"Converting text -> mel for {uploaded.name}...")
                text_clean = clean_text(text)
                seq = text_to_sequence(text_clean)
                # create emotion vector
                emotion_vector = np.array([happy, sad, angry, neutral], dtype=np.float32)
                mel = infer_taco(taco_m, seq, emb, emotion_vector=emotion_vector)
                # status.info(f"Vocoder (mel -> waveform) for {uploaded.name}...")
                wav_out = infer_voc(voc_m, mel)
                wav_out = normalize_audio(wav_out)
                
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Ensure unique filename even if processing happens in same second
                out_path = f"tmp/output_{timestamp}_{i}.wav"
                save_wav(out_path, wav_out, sr=16000)
                st.audio(out_path)
                st.download_button(f"Download Audio ({uploaded.name})", data=open(out_path,"rb").read(), file_name=f"output_{timestamp}_{i}.wav", mime="audio/wav", key=f"dl_{timestamp}_{i}")
            
            status.success("Batch processing complete.")
