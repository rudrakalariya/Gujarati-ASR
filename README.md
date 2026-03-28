<h1 align="center">🎙️ Gujarati Custom ASR Pipeline</h1>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Hosted_Model-orange)
![UI](https://img.shields.io/badge/UI-Gradio-brightgreen)

</div>

## 📖 Abstract / Introduction
Welcome to the Custom Two-Stage Automatic Speech Recognition (ASR) Pipeline exclusively fine-tuned for the **Gujarati language**. 

Transcribing Gujarati presents unique challenges, including handling complex phonetic overlapping, messy Unicode rendering, and disconnected language matras. This project solves these challenges using a powerful hybrid architecture:
1. **Acoustic Base:** A Whisper-Small model fine-tuned on the *Kathbath dataset* to recognize domain-specific Gujarati speech.
2. **Linguistic Post-Processing:** Integration of the Smruti API / IndicNLP for grammar correction and robust text normalization.

Whether you're dealing with noisy environments or complex sentence structures, this pipeline produces highly accurate transcriptions while ensuring text is grammatically structured and uniformly represented.

---

## 🏗️ Architecture
Our pipeline systematically processes audio and text across two distinct stages:

### Stage 1: The Acoustic Model
Built upon Hugging Face's `pipeline`, our architecture inherently leverages smart pre-processing features:
* **Auto-16kHz Resampling:** The system intelligently forces any incoming microphone/uploaded audio to 16kHz, completely preventing the notorious "alien noise" transcription distortion.
* **30-Second Chunking with Overlap:** Solves major memory bottlenecks. We cut long audio inputs into safe 30s chunks with a 5-second overlapping stride, guaranteeing that words are never split awkwardly during transcription transitions.

### Stage 2: The Gujarati Text Gate (Post-Processing)
Once the raw transcribed text is generated, it passes through our linguistic normalization gateway. By integrating IndicNLP rules, the system repairs messy Unicode sequences effectively, ensuring all dependent characters/matras attach properly to their base consonants.

---

## 🚀 Results & Performance
Our two-stage approach prominently enhances the overall transcription quality. Below are the definitive Word Error Rates (WER):

| Model Stage / Configuration | Word Error Rate (WER) |
| :--- | :---: |
| **Baseline** *(Direct Whisper-Small Model)* | `36.0%` |
| **Stage 1 & 2 Fine-Tuned Model** *(Domain Adaptation on Kathbath Dataset)* | `27.0%` |
| **Final Hybrid Pipeline** *(Model + Linguistic Post-Processing)* | **`16.0%`** 🎉 |

---

## 📂 Directory Structure

To maintain a clean codebase for deployment, all experimental and training notebooks are located in a dedicated `notebooks/` folder.

```text
📦 Gujarati-ASR/
 ┣ 📂 notebooks/
 ┃ ┣ 📜 Phase1 Training.ipynb   # Baseline evaluation & initial structural fine-tuning
 ┃ ┗ 📜 Phase 2 Training.ipynb  # Stage 2 Domain Adaptation (Kathbath dataset)
 ┣ 📜 app.py                    # Main pipeline deployment & Gradio Web Interface
 ┣ 📜 generate_dataset.py       # Data collation and preprocessing routines
 ┣ 📜 test_fleurs.py            # Evaluation logic on FLEURS datasets
 ┣ 📜 test_kathbath.py          # Domain-specific testing on the Kathbath dataset
 ┣ 📜 test_wer.py               # Calculation routines for plotting Word Error Rate metrics
 ┣ 📜 test_with_smruti.py       # Standalone scripts testing grammar/linguistic correction
 ┣ 📜 requirements.txt          # Minimal Python dependencies for inference
 ┗ 📜 README.md                 # Project Documentation

## 🧠 Model Hosting
The final, fully fine-tuned acoustic model weights are approximately 2.7 GB. Due to GitHub's file size limits, these weights are not stored in this repository.

Instead, we have made the model publicly accessible and permanently hosted on the Hugging Face Hub. Our deployment script, app.py, is pre-configured to automatically pull and cache these exact model weights from Hugging Face during its first run using the Hugging Face pipeline function.

🔗 [Click here to view, download, or test the Model on Hugging Face](https://huggingface.co/rudrakalariya/Gujarati-ASR)

### Downloading the Model Manually
If you prefer to download the model weights manually from Hugging Face, follow these steps:

**Method 1: Using Hugging Face CLI (Recommended)**
1. Install the Hugging Face Hub CLI:
   ```bash
   pip install -U "huggingface_hub[cli]"
   ```
2. Download the model to a local directory (e.g., `model/`):
   ```bash
   huggingface-cli download https://huggingface.co/rudrakalariya/Gujarati-ASR --local-dir model/
   ```

**Method 2: Using Git LFS**
1. Ensure you have Git Large File Storage (LFS) installed:
   ```bash
   git lfs install
   ```
2. Clone the model repository directly:
   ```bash
   git clone https://huggingface.co/https://huggingface.co/rudrakalariya/Gujarati-ASR
   ```


💻 How to Run / Installation
You can easily get this ASR pipeline up and running locally to test audio files or speak directly through your microphone.

1. Clone the Repository
bash
git clone https://github.com/rudrakalariya/Gujarati-ASR.git
cd Gujarati-ASR
2. Install Dependencies
Make sure you have Python installed, then install the required inference packages. (Make sure PyTorch is installed based on your environment).

bash
pip install -r requirements.txt
3. Launch the Application
Run the main web interface file. The system will handle downloading the Hugging Face model and initializing the post-processing engine.

bash
python app.py
Once the model weights are fully loaded, your terminal will provide a local address (usually http://127.0.0.1:7860). Open that URL in your browser to start transcribing!
