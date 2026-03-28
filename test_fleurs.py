from datasets import load_dataset
from transformers import pipeline
from transformers import logging as hf_logging
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from jiwer import wer
import warnings

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# 1. THE GATE
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("gu")

def clean_text(text):
    if not isinstance(text, str): return ""
    return normalizer.normalize(text)

# 2. LOAD YOUR MODEL
print("System: Loading Model...")
asr_system = pipeline(
    "automatic-speech-recognition",
    model="./gujarati_final_model",
    device="cuda", 
    batch_size=2, # Fast batched processing
    generate_kwargs={"language": "gujarati", "task": "transcribe"} 
)

MY_HF_TOKEN = "hf_your_token_here"  # Replace with your actual Hugging Face token
# 3. PULL GOOGLE FLEURS DIRECTLY FROM THE CLOUD
# "gu_in" is the code for Gujarati (India). We only pull the "test" split.
print("System: Downloading Google FLEURS (Gujarati Human Test Set)...")
fleurs_test = load_dataset("google/fleurs", "gu_in", split="test", trust_remote_code=True, token=MY_HF_TOKEN).select(range(50))  # Limit to 50 samples for quick testing

predictions = []
references = []

print(f"\n🚀 Testing on {len(fleurs_test)} real human audio files...")

# 4. FAST EVALUATION LOOP
# The dataset library automatically handles the audio arrays, no .wav files needed!
try:
    # Extract the raw audio arrays to feed to the model
    audio_inputs = [sample["audio"]["array"] for sample in fleurs_test]
    actual_texts = [sample["transcription"] for sample in fleurs_test]

    # Run the model
    results = asr_system(audio_inputs)
    
    for i, out in enumerate(results):
        hypothesis = clean_text(out["text"])
        reference = clean_text(actual_texts[i])
        
        predictions.append(hypothesis)
        references.append(reference)
        
        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(fleurs_test)} human sentences...")
            
except Exception as e:
    print(f"Error during processing: {e}")

# 5. CALCULATE HUMAN WER
print("\nCalculating Final Human Score...")
final_wer = wer(references, predictions)

print(f"=====================================")
print(f"🎤 FINAL WER (GOOGLE FLEURS): {final_wer * 100:.2f}%")
print(f"=====================================")