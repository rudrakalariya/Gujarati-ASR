import gradio as gr
from transformers import pipeline
from transformers import logging as hf_logging 
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import warnings

# Suppress annoying terminal warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# --- 1. THE GUJARATI TEXT GATE (Post-Processing) ---
# Solves Issue: Messy Unicode and disconnected Matras
print("System: Initializing Gujarati Normalization Gate...")
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("gu")

def clean_gujarati_output(text):
    if not text: return ""
    return normalizer.normalize(text)

# --- 2. LOAD THE AI PIPELINE ---
# Solves Issue: 16kHz resampling and 30-second chunking
MODEL_PATH = "./gujarati_phase2_model"  
print("System: Loading Neural Network Weights...")

# The Hugging Face 'pipeline' is brilliant. It automatically forces any incoming 
# microphone audio into 16kHz, preventing the "alien noise" issue.
asr_system = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH,
    device="cuda",         
    chunk_length_s=30,     # Solves the Memory Issue: Cuts long audio into safe 30s chunks!
    stride_length_s=5,
    ignore_warning=True, 
    generate_kwargs={"language": "gujarati", "task": "transcribe"}   # Creates a 5-second overlap between chunks so words aren't cut in half
)

# --- 3. THE INFERENCE FUNCTION ---
def transcribe_audio(audio_filepath):
    if audio_filepath is None:
        return "Please record or upload an audio file."
    
    try:
        # Step A: Feed the audio to the AI
        raw_output = asr_system(audio_filepath)
        raw_text = raw_output["text"]
        
        # Step B: Pass the AI's answer through the Gate
        final_clean_text = clean_gujarati_output(raw_text)
        
        return final_clean_text
    except Exception as e:
        return f"An error occurred during transcription: {str(e)}"

# --- 4. THE USER INTERFACE ---
print("System: Building Web Interface...")
ui = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Speak or Upload Gujarati"),
    outputs=gr.Textbox(label="📝 Transcribed Gujarati Text", lines=5),
    title="Gujarati Speech-to-Text",
    description="Custom Whisper Model | Validation Loss: ~0.21 | Features: Auto-16kHz Resampling, 30s Chunking, IndicNLP Normalization.",
    theme="soft"
)
ui.queue()  # Enable queuing for better performance with multiple users

# Launch the app!
if __name__ == "__main__":
    print("✅ System Ready! Open the local URL provided below.")
    ui.launch()