import sys
import os
from pathlib import Path
import numpy as np
import streamlit as st

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.utils.embedding_manager import EmbeddingManager
from src.utils.speaker_embedding_cache import get_or_compute_embedding_from_file, get_or_compute_embedding_from_bytes
from src.utils.device import get_device
from src.inference.speaker_encoder import load_model as load_spk, infer as infer_spk
from src.inference.tacotron2_infer import load_model as load_taco, infer as infer_taco
from src.inference.vocoder_infer import load_model as load_voc, infer as infer_voc
from src.audio.audio_utils import load_wav, save_wav, normalize_audio, extract_mel
from src.text.text_cleaner import clean_text, text_to_sequence

st.set_page_config(page_title="Voice Clone", layout="wide", page_icon="🎙️")

st.title("🎙️ Voice Clone with Speaker Library")

# Initialize manager globally
manager = EmbeddingManager(root_dir="cache/embeddings")

# Sidebar: device / model options
with st.sidebar:
    st.header("⚙️ Settings")
    device_choice = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    half = st.checkbox("Use half precision (fp16) for speed", value=False)
    
    st.markdown("**Models**")
    spk_path = st.text_input("Speaker encoder path", "models/speaker_encoder.pt")
    taco_path = st.text_input("Tacotron2 path", "models/tacotron2.pt")
    voc_path = st.text_input("Vocoder path", "models/vocoder.pt")

    st.markdown("---")
    st.markdown("## 🎙️ Speaker Library")

    speakers = manager.list_speakers()
    speaker_options = [f"{s['id']} — {s['name']}" for s in speakers]
    speaker_ids = [s["id"] for s in speakers]

    selected_speaker_id = None
    if speakers:
        choice = st.selectbox("Select saved speaker", ["(None)"] + speaker_options)
        if choice != "(None)":
            idx = speaker_options.index(choice)
            selected_speaker_id = speaker_ids[idx]
    else:
        st.info("No speakers saved yet.")

    st.markdown("---")
    st.markdown("### ➕ Save current speaker")

    new_speaker_id = st.text_input("Speaker ID (no spaces)", placeholder="e.g. rahul_happy")
    new_speaker_name = st.text_input("Display Name", placeholder="Rahul (Happy tone)")
    new_speaker_notes = st.text_area("Notes (optional)", height=60)

    save_btn = st.button("Save current embedding")

    if save_btn:
        embedding = st.session_state.get("current_embedding")
        if embedding is None:
            st.error("No embedding found. Generate once before saving.")
        else:
            if not new_speaker_id.strip():
                st.error("Please provide a Speaker ID.")
            else:
                try:
                    manager.save_embedding(
                        speaker_id=new_speaker_id.strip(),
                        embedding=np.asarray(embedding, dtype=np.float32),
                        name=new_speaker_name.strip() or None,
                        notes=new_speaker_notes.strip() or ""
                    )
                    st.success(f"Saved speaker as '{new_speaker_id.strip()}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save embedding: {e}")

    st.markdown("---")
    st.markdown("### ✏️ Manage selected speaker")

    if selected_speaker_id:
        new_name = st.text_input("Rename display name", key="rename_name")
        if st.button("Rename"):
            if new_name.strip():
                manager.rename_speaker(selected_speaker_id, new_name.strip())
                st.success("Speaker renamed. Refresh to see changes.")
                st.rerun()
            else:
                st.error("Display name cannot be empty.")

        if st.button("🗑️ Delete speaker"):
            manager.delete_speaker(selected_speaker_id)
            st.success("Speaker deleted.")
            st.rerun()
    else:
        st.caption("Select a speaker above to rename or delete.")

# Load models (cached)
@st.cache_resource
def load_models_cached(spk, taco, voc, user_device, half_flag):
    if user_device == "auto":
        prefer_gpu = True
    else:
        prefer_gpu = (user_device == "cuda")
        
    device_obj = get_device(prefer_gpu=prefer_gpu)
    
    spk_m = taco_m = voc_m = None
    errors = []
    
    try: spk_m = load_spk(spk, device_obj, half_flag)
    except Exception as e: errors.append(f"Speaker encoder: {e}")
    
    try: taco_m = load_taco(taco, device_obj, half_flag)
    except Exception as e: errors.append(f"Tacotron2: {e}")
    
    try: voc_m = load_voc(voc, device_obj, half_flag)
    except Exception as e: errors.append(f"Vocoder: {e}")
    
    return spk_m, taco_m, voc_m, device_obj, errors

# Main UI
col1, col2 = st.columns([1,1])
with col1:
    st.header("1️⃣ Input")
    input_method = st.radio("Input Method", ["Upload File", "Record Voice"])
    
    uploaded_files = []
    if input_method == "Upload File":
        files = st.file_uploader("Upload reference voice (wav/mp3)", type=["wav","mp3"], accept_multiple_files=True)
        if files:
            uploaded_files = files
    else:
        if "audio_key" not in st.session_state:
            st.session_state.audio_key = 0
            
        audio_value = st.audio_input("Record your voice", key=f"audio_recorder_{st.session_state.audio_key}")
        if audio_value:
            uploaded_files = [audio_value]
            
            # Save functionality
            st.markdown("### 💾 Save Recording")
            col_save_1, col_save_2 = st.columns([3, 1])
            with col_save_1:
                save_name = st.text_input("Filename (optional)", placeholder="my_recording")
            with col_save_2:
                st.write("") # Spacer
                st.write("") # Spacer
                if st.button("Save to Disk"):
                    try:
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        if save_name.strip():
                            filename = f"{save_name.strip()}.wav"
                        else:
                            filename = f"recording_{timestamp}.wav"
                            
                        save_dir = Path("recordings")
                        save_dir.mkdir(exist_ok=True)
                        save_path = save_dir / filename
                        
                        with open(save_path, "wb") as f:
                            f.write(audio_value.getvalue())
                        
                        st.success(f"Saved to {save_path}")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
            
        if st.button("Delete Recording"):
            st.session_state.audio_key += 1
            st.rerun()

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) ready.")

    st.markdown("### 🗣 Text")
    text = st.text_area("Text to synthesize", value="Hello, this is a cloned voice.", height=150)
    
    st.markdown("### 🎭 Emotions")
    happy = st.slider("Happy", 0.0, 1.0, 0.0)
    sad = st.slider("Sad", 0.0, 1.0, 0.0)
    angry = st.slider("Angry", 0.0, 1.0, 0.0)
    neutral = st.slider("Neutral", 0.0, 1.0, 1.0)
    
    generate = st.button("🚀 Generate")

with col2:
    st.header("2️⃣ Output")
    status = st.empty()
    
    if generate:
        if not uploaded_files and not selected_speaker_id:
            st.error("Please upload reference voice OR select a saved speaker.")
        elif not text.strip():
            st.error("Please enter text.")
        else:
            status.info("Loading models...")
            spk_m, taco_m, voc_m, device_obj, errors = load_models_cached(spk_path, taco_path, voc_path, device_choice, half)
            
            if errors:
                for e in errors: st.error(e)
            else:
                # If no files uploaded but speaker selected, create a dummy list to trigger loop once
                # But the loop logic below relies on 'uploaded' for name. 
                # Let's handle the "Saved Speaker Only" case.
                
                # If we have uploaded files, we process them.
                # If we DON'T have uploaded files but HAVE a selected speaker, we run once.
                
                loop_targets = uploaded_files if uploaded_files else [None]
                
                for i, uploaded in enumerate(loop_targets):
                    display_name = uploaded.name if uploaded else f"Saved Speaker ({selected_speaker_id})"
                    st.markdown(f"### Processing: {display_name}")
                    
                    emb = None
                    
                    # 1. Try loading saved speaker first if selected
                    if selected_speaker_id:
                        try:
                            emb = manager.load_embedding(selected_speaker_id)
                            st.info(f"Using saved speaker: {selected_speaker_id}")
                        except Exception as e:
                            st.error(f"Failed to load saved speaker: {e}")
                    
                    # 2. If no saved speaker (or failed), compute from upload
                    if emb is None:
                        if uploaded:
                            audio_bytes = uploaded.getvalue()
                            emb = get_or_compute_embedding_from_bytes(
                                audio_bytes=audio_bytes,
                                sr=16000,
                                spk_model=spk_m,
                                infer_fn=infer_spk,
                                cache_dir="cache/embeddings"
                            )
                        else:
                            st.error("No reference audio provided and no valid saved speaker selected.")
                            continue

                    # Store for saving
                    st.session_state.current_embedding = emb
                    
                    # Text -> Seq
                    # Explicit cleaning as requested
                    text_clean = clean_text(text)
                    # use_phonemes=False for VCTK
                    seq = text_to_sequence(text_clean, use_phonemes=False)
                    
                    emotion_vector = np.array([happy, sad, angry, neutral], dtype=np.float32)
                    
                    # Tacotron
                    mel = infer_taco(taco_m, seq, emb, emotion_vector=emotion_vector)
                    
                    # Vocoder
                    wav_out = infer_voc(voc_m, mel)
                    wav_out = normalize_audio(wav_out)
                    
                    # Save & Play
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = f"tmp/output_{timestamp}_{i}.wav"
                    Path("tmp").mkdir(exist_ok=True)
                    save_wav(out_path, wav_out, sr=16000)
                    
                    st.audio(out_path)
                    st.download_button("Download", data=open(out_path,"rb").read(), file_name=f"output_{i}.wav", mime="audio/wav", key=f"dl_{i}")
                    
            status.success("Done!")
