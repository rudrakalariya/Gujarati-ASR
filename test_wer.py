import pandas as pd
from transformers import pipeline
from transformers import logging as hf_logging 
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from jiwer import wer
import warnings
import os


# Mute warnings for a clean terminal
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# --- 1. SETUP THE GATE ---
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("gu")

def clean_text(text):
    if not isinstance(text, str): return ""
    return normalizer.normalize(text)

# --- 2. LOAD THE MODEL ---
print("Loading Model for Testing...")
asr_system = pipeline(
    "automatic-speech-recognition",
    model="./gujarati_phase2_model",  
    device="cuda", 
    chunk_length_s=30,
    stride_length_s=5,
    ignore_warning=True,
    generate_kwargs={"language": "gujarati", "task": "transcribe"} 
)

# --- 3. LOAD YOUR TEST DATASET ---
# Change this to the name of your actual test dataset file
print("Loading Test Dataset...")
df = pd.read_csv("./dataset500/dataset.csv")

predictions = []
references = []

print(f"Starting evaluation on {len(df)} audio files...")

# --- 4. THE TESTING LOOP ---
AUDIO_FOLDER = "./dataset500/audio/"  # Change this to your actual audio folder path
for index, row in df.head(250).iterrows():
    file_name = row['audio_file'] # The column with the audio file name
    audio_path = os.path.join(AUDIO_FOLDER, str(file_name))
    actual_text = row['ground_truth']   # The column with the correct human translation
    
    try:
        # A. Get the AI's prediction
        raw_output = asr_system(audio_path)
        
        # B. Pass both through the Gate to ensure fair comparison
        hypothesis = clean_text(raw_output["text"])
        reference = clean_text(actual_text)
        
        predictions.append(hypothesis)
        references.append(reference)
        
        # Print progress every 10 files
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{len(df)} files...")
            
    except Exception as e:
        print(f"Skipping {audio_path} due to error: {e}")

# --- 5. CALCULATE THE FINAL WER ---
print("\nCalculating Final Score...")
final_wer = wer(references, predictions)

# Convert to a readable percentage
print(f"=====================================")
print(f"🎉 FINAL WORD ERROR RATE: {final_wer * 100:.2f}%")
print(f"=====================================")