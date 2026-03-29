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

---

## 📂 Project Structure

```
Gujarati-ASR/
│
├── gujarati_phase2_model/        # Fine-tuned ASR model (Stage 2 checkpoints)
│
├── notebooks/                   # Training & experimentation notebooks
│   ├── Phase1_Training.ipynb    # Baseline model training & evaluation
│   └── Phase2_Training.ipynb    # Domain adaptation (Kathbath dataset)
│
├── app.py                       # Main application (Gradio UI + ASR pipeline)
├── generate_dataset.py          # Dataset preparation & preprocessing
│
├── test_fleurs.py               # Evaluation on FLEURS dataset
├── test_kathbath.py             # Evaluation on Kathbath dataset
├── test_wer.py                  # Word Error Rate (WER) calculation
├── test_with_smruti.py          # Testing linguistic post-processing
│
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 🧠 Model Hosting
The final, fully fine-tuned acoustic model weights are approximately 2.7 GB. Due to GitHub's file size limits, these weights are not stored in this repository.

Instead, we have made the model publicly accessible and permanently hosted on the Hugging Face Hub. Our deployment script, app.py, is pre-configured to automatically pull and cache these exact model weights from Hugging Face during its first run using the Hugging Face pipeline function.

🔗 [Click here to view, download, or test the Model on Hugging Face](https://huggingface.co/rudrakalariya/Gujarati-ASR)

---

## 💻 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/rudrakalariya/Gujarati-ASR.git
cd Gujarati-ASR
```

---

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download Model from Hugging Face
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download rudrakalariya/Gujarati-ASR --local-dir gujarati_phase2_model/
```

---

### 5️⃣ Run the Application
```bash
python app.py
```

---
