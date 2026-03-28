import pandas as pd
import requests
import os
from transformers import pipeline
from transformers import logging as hf_logging 
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from jiwer import wer
import warnings

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("gu")

def clean_text(text):
    if not isinstance(text, str): return ""
    return normalizer.normalize(text)

print("System: Loading Whisper ASR Model into GPU...")
asr_system = pipeline(
    "automatic-speech-recognition",
    model="./gujarati_phase2_model",
    device="cuda", 
    chunk_length_s=30,
    stride_length_s=5,
    ignore_warning=True,
    generate_kwargs={"language": "gujarati", "task": "transcribe"} 
)

print("System: Loading Test Dataset...")
df = pd.read_csv("./dataset500/dataset.csv")
AUDIO_FOLDER = "./dataset500/audio/"

# We will store the temporary results here
gpu_results = []
predictions = []
references = []

# ==========================================
# PHASE 1: THE GPU SPRINT (100% Utilization)
# ==========================================
print("\n🚀 PHASE 1: Running GPU Inference...")
for index, row in df.head(50).iterrows():
    file_name = row['audio_file'] 
    audio_path = os.path.join(AUDIO_FOLDER, str(file_name))
    actual_text = row['ground_truth']   
    
    try:
        # The GPU does the math instantly and moves to the next file without waiting!
        raw_output = asr_system(audio_path)
        whisper_text = raw_output["text"]
        
        # Save it to our temporary list
        gpu_results.append({
            "index": index,
            "whisper_text": whisper_text,
            "actual_text": actual_text
        })
        
        if (index + 1) % 10 == 0:
            print(f"GPU Processed {index + 1}/50 audio files...")
    except Exception as e:
        print(f"Skipping {audio_path} due to ASR error: {e}")

# ==========================================
# PHASE 2: THE API NETWORK CALLS
# ==========================================
print("\n📡 PHASE 2: Sending to Smruti Language Model...")
for item in gpu_results:
    whisper_text = item["whisper_text"]
    actual_text = item["actual_text"]
    corrected_text = whisper_text # Default fallback
    
    try:
        response = requests.post(
            "https://vrund1346-smruti-gujarati-grammar-checker.hf.space/correct",
            json={"sentence": whisper_text},
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        if response.status_code == 200:
            data = response.json()
            corrected_text = data.get("corrected_text", whisper_text)
        else:
            print(f"⚠️ API error {response.status_code} on row {item['index']}")
    except Exception as e:
        print(f"❌ API failed on row {item['index']}: {e}")

    # Pass through the Gate and save for WER calculation
    predictions.append(clean_text(corrected_text))
    references.append(clean_text(actual_text))
    
    if (item['index'] + 1) % 10 == 0:
        print(f"API Processed {item['index'] + 1}/50 sentences...")

# ==========================================
# PHASE 3: CALCULATE METRICS
# ==========================================
print("\nCalculating Final Score...")
final_wer = wer(references, predictions)

print(f"=====================================")
print(f"🎉 FINAL WORD ERROR RATE: {final_wer * 100:.2f}%")
print(f"=====================================")