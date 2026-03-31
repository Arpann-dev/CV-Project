# Document Text Extractor

A computer vision project that extracts text from image-based documents using an OpenCV preprocessing pipeline and EasyOCR's deep learning backend (CRAFT + CRNN), served through a Streamlit web interface.

---
LIVE DEMO: https://document-text-extractor.streamlit.app/
## How It Works

**Stage 1 — CV Preprocessing (OpenCV)**
- Grayscale conversion
- Adaptive denoising — Gaussian blur (light noise) or NL-Means (heavy noise), auto-selected by pixel standard deviation
- Deskewing — Hough Line Transform estimates tilt angle and corrects it via affine rotation
- Binarization — Otsu or Adaptive thresholding
- Morphological cleanup and edge enhancement (optional)

**Stage 2 — Deep Learning OCR (EasyOCR)**
- CRAFT (VGG-16 CNN) detects text regions via character region and affinity score maps
- CRNN (CNN + BiLSTM + CTC) recognizes text within each detected region
- Runs on GPU automatically if CUDA is available via PyTorch

**Stage 3 — Post-processing**
- Confidence threshold filtering
- Reading order sort (Y-band grouping → left-to-right per line)

---

## Features

- Supports JPG, PNG, BMP, TIFF inputs
- Toggle each preprocessing step independently from the sidebar
- Colour-coded bounding box overlay (green = high confidence, yellow = low)
- Batch processing — upload multiple images at once
- Export extracted text as `.txt` or `.json`
- GPU acceleration auto-detected

---

## Setup

**Requirements**
- Python 3.10+
- NVIDIA GPU with CUDA 11.8 (optional but recommended)

**Install dependencies**
```bash
pip install streamlit opencv-python easyocr torch torchvision numpy pillow
```

**Run the app**
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Project Structure

```
CV-Project/
├── app.py          # Main Streamlit application
└── README.md
```

---

## Tech Stack

| Component | Library |
|---|---|
| Web interface | Streamlit |
| Computer vision | OpenCV |
| Text detection | EasyOCR (CRAFT) |
| Text recognition | EasyOCR (CRNN) |
| Deep learning backend | PyTorch |

---

## Course

Computer Vision Project — CSE-48, KIIT University, March 2026  
Submitted to Dr. Rinku Datta Rakshit
