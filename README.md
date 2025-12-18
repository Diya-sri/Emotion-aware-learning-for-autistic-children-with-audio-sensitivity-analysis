#  Emotion-Aware Learning for Autistic Children with Audio Sensitivity Analysis

An interactive, research-driven web application designed to support children with Autism Spectrum Disorder (ASD) by combining **real-time emotion recognition**, **emoji-based learning**, and **audio sensitivity analysis**. The system supports structured emotional learning, tracks response patterns, and provides automated caregiver insights.

---

##  Abstract

This project introduces a web-based platform that assists autistic children in identifying emotions by mapping real-time facial expressions of parents into simple, intuitive emojis. Children respond through guided attempts, allowing measurable emotional learning progress.

The system also analyzes reactions to uploaded audio samples to detect potential auditory discomfort triggers—helping understand sensory sensitivity patterns. The platform generates a parental report summarizing performance, emotional learning, and observed sensitivity responses.

---

##  Project Overview

This repository contains the full implementation and methodology of the system, integrating:

- **Computer Vision-based emotion detection**
- **Deep learning for sentiment classification**
- **Real-time interaction workflows**
- **Audio response tracking**
- **Automated reporting for caregivers**

The project serves as a proof-of-concept for technology-assisted emotional development in autistic children.

---

##  Dataset Information

**Dataset Used:** FER-2013 (Facial Expression Recognition)  
 Total Images: **35,887**  
 Resolution: **48×48 pixels (grayscale)**  
 Labels: 7 emotion classes  

Download from Kaggle via:

```bash
python scripts/download_dataset.py



##  Methodology

**###  1. Environment Setup**

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

###  2. Data Acquisition & Preprocessing

```bash
python scripts/download_dataset.py
python src/preprocessing/preprocess_data.py
```

Includes data cleaning, normalization, and augmentation.

---

###  3. Model Training & Evaluation

**Train Emotion Recognition Model:**

```bash
python src/training/train_model.py --epochs 50 --batch_size 32
```

**Evaluate Performance:**

```bash
python src/evaluation/evaluate.py
```

* **Model Backbone:** InceptionV3 (Transfer Learning)
* **Classifier:** Fine-tuned for FER-2013 emotion classes

---

##  System Architecture

| Module             | Description                                                         |
| ------------------ | ------------------------------------------------------------------- |
| Data Preprocessing | Normalization, augmentation, dataset formatting                     |
| Model Training     | InceptionV3-based emotion classifier                                |
| Learning Engine    | Converts detected emotion → emoji for structured emotional learning |
| Response Tracking  | Logs repeated attempts, improvements & behavioral metrics           |
| Audio Sensitivity  | Detects reaction patterns to uploaded sound stimuli                 |
| Reporting          | Generates caregiver report with learning progress insights          |

---

## ✔️ Current Results

* ✓ Real-time facial emotion detection working
* ✓ Multi-attempt learning workflow tested
* ✓ Report generation and response tracking implemented
* ✓ FER-2013 validation confirms strong recognition accuracy

---

##  Dependencies

Defined in **requirements.txt**

Core libraries:

* TensorFlow / Keras
* NumPy / Pandas
* OpenCV
* Flask
* Matplotlib
* scikit-learn

---

##  Repository Structure

```
Emotion-Aware-Learning/
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
│
├── scripts/
├── data/         # ignored in git
├── results/      # generated outputs
├── requirements.txt
└── README.md
```

---

##  .gitignore Rules

```
__pycache__/
*.pyc
*.h5
*.pkl
*.zip
data/
models/
results/
.env
```

---

##  References

* Rashidan et al. (2021). *Technology-Assisted Emotion Recognition for ASD*. IEEE Access.
* Chien et al. (2023). *Game-Based Cognitive Assessment with Eye Tracking*. IEEE Transactions.
* Bartl-Pokorny et al. (2021). *Robot-Based Autism Intervention*. IEEE Access.
* Prakash et al. (2023). *Computer Vision for Autism Assessment*. IEEE Access.
* Kurian & Tripathi (2025). *mAutNet: Multimodal Emotion Recognition in ASD*. IEEE Access.

---

## Future Scope

* Integrate Transformer-based emotion analysis
* Add multilingual support
* Optimize model for deployment on edge devices (Jetson Nano / Mobile)

---


