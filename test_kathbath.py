import torch
from datasets import load_dataset
from transformers import pipeline
from transformers import logging as hf_logging
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from jiwer import wer
import warnings

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# --- 1. THE GATE ---
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("gu")

def clean_text(text):
    if not isinstance(text, str): return ""
    return normalizer.normalize(text)

# --- 2. LOAD YOUR MODEL IN TURBO MODE ---
print("System: Loading Model in FP16 Turbo Mode...")
asr_system = pipeline(
    "automatic-speech-recognition",
    model="./gujarati_final_model",
    device="cuda", 
    torch_dtype=torch.float16, # Turbo Mode Enabled!
    batch_size=4,              # High-speed batch processing
    generate_kwargs={"language": "gujarati", "task": "transcribe"} 
)

# --- 3. PULL AI4BHARAT KATHBATH (50 SENTENCES) ---
# Replace this with your actual Hugging Face token
MY_HF_TOKEN = "hf_your_token_here"  # Replace with your actual Hugging Face token

print("System: Downloading AI4Bharat Kathbath (Gujarati Human Test Set)...")
kathbath_test = load_dataset(
    "ai4bharat/Kathbath", 
    "gujarati",                # Kathbath uses the full language name
    split="valid", 
    trust_remote_code=True,
    token=MY_HF_TOKEN
).select(range(50)) 

predictions = []
references = []

print(f"\n🚀 Testing on {len(kathbath_test)} real human audio files on Home Turf...")

# --- 4. FAST EVALUATION LOOP ---
try:
    # Extract audio arrays
    audio_inputs = [sample["audio_filepath"]["array"] for sample in kathbath_test]
    
    # Kathbath stores the human text in the "sentence" column
    actual_texts = [sample["text"] for sample in kathbath_test]

    # Run the model
    results = asr_system(audio_inputs)
    
    for i, out in enumerate(results):
        hypothesis = clean_text(out["text"])
        reference = clean_text(actual_texts[i])
        
        predictions.append(hypothesis)
        references.append(reference)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/50 human sentences...")
            
except Exception as e:
    print(f"Error during processing: {e}")

# --- 5. CALCULATE HUMAN WER ---
print("\nCalculating Final Home Turf Score...")
final_wer = wer(references, predictions)

print(f"=====================================")
print(f"🎤 FINAL WER (KATHBATH): {final_wer * 100:.2f}%")
print(f"=====================================")