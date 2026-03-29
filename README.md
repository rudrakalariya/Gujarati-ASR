<h1 align="center">🎙️ Gujarati Custom ASR Pipeline</h1>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Hosted_Model-orange)
![UI](https://img.shields.io/badge/UI-Gradio-brightgreen)

</div>

---

## 📖 Introduction
This project is a **Custom Two-Stage Automatic Speech Recognition (ASR) Pipeline** designed specifically for the **Gujarati language**.

Gujarati ASR is challenging due to:
- Complex phonetics
- Unicode inconsistencies
- Matra (vowel sign) attachment issues

This pipeline solves these problems using:
1. 🎧 **Fine-tuned Whisper Model** (Acoustic Understanding)
2. 🧹 **IndicNLP Normalization** (Text Cleaning & Structuring)

---

## 🏗️ Architecture

### 🔹 Stage 1: Acoustic Model
- Whisper-based ASR (fine-tuned)
- Auto 16kHz audio resampling
- Chunking (30s with overlap)
- Handles long audio efficiently

### 🔹 Stage 2: Gujarati Text Normalization
- Uses Indic NLP
- Fixes Unicode issues
- Corrects matra placement
- Produces clean Gujarati text

---

## 🚀 Performance

| Model Stage | WER |
|------------|-----|
| Baseline Whisper | 36% |
| Fine-tuned Model | 27% |
| Final Pipeline | **16% 🎉** |

---

## 📂 Project Structure

```text
📦 Gujarati-ASR/
 ┣ 📂 gujarati_phase2_model/
 ┣ 📂 notebooks/
 ┣ 📜 app.py
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

---

## 🧠 Model Hosting

The trained model (~2.7GB) is hosted on Hugging Face:

🔗 https://huggingface.co/rudrakalariya/Gujarati-ASR

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

## 🌐 Usage

- Open browser at: `http://127.0.0.1:7860`
- Upload or record Gujarati audio
- Get clean Gujarati transcription instantly

---

## ✨ Features

- 🎤 Microphone + File Upload
- 🤖 Custom Gujarati ASR Model
- 🧹 Indic NLP Text Cleaning
- ⚡ Chunked Processing (handles long audio)
- 🌐 Simple Gradio UI

---

## 📌 Notes

- First run may take time (model loading)
- CPU mode supported (GPU optional)
- Ensure model folder exists (`gujarati_phase2_model/`)

---

## 🚀 Future Improvements

- Real-time streaming transcription
- Gujarati → English translation
- Mobile app integration
- Faster inference optimization

---

## 🤝 Contribution

👨‍💻 Developed by:

- **[Rudra Kalariya](https://github.com/rudrakalariya)**
- **[Kalpesh Gangani](https://github.com/kalpeshgangani16)**
- **[Preya Dhangar](https://github.com/Itz-preya)**

Feel free to contribute, open issues, or suggest improvements!

## 📜 License

This project is for educational and research purposes.
