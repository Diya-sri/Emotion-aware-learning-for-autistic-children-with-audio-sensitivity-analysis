# Emotion-aware-learning-for-autistic-children-with-audio-sensitivity-analysis
Emotion-Aware Learning for Autistic Children with Audio Sensitivity Analysis
Abstract:
This project introduces a web-based application developed to assist children with autism in recognizing emotions and managing sensitivity to various sounds. The system employs a deep learning model (InceptionV3) trained on the FER‑2013 facial expression dataset to classify parental emotions in real time. These detected emotions are then represented through emojis on the child’s interface to facilitate emotional learning and engagement. The system also monitors each child’s reactions during multiple interaction attempts, enabling structured progress tracking. An integrated sound analysis module records facial responses to different audio frequencies to identify potential triggers. Automated reports summarize the child’s emotional and sensory development over time, helping caregivers and educators tailor interventions for personalized support.

Project Overview:
This repository contains all source code and documentation for a research-oriented web application aimed at improving social-emotional learning among children with Autism Spectrum Disorder (ASD). The platform combines facial emotion recognition with audio sensitivity analysis to provide real-time emotional feedback and structured behavioral monitoring. By integrating AI-based recognition with visual and auditory learning cues, the project supports autism research, inclusive education, and parent-assisted emotional development.

Dataset Used:
Dataset: FER‑2013 (Facial Expression Recognition)

Description: Contains 35,887 grayscale images (48 × 48 pixels), labeled into seven categories: anger, disgust, fear, happiness, sadness, surprise, and neutral.

Preprocessing: Each image undergoes normalization, resizing, and augmentation to improve generalization during training.

Automatic download:

bash
python scripts/download_dataset.py
Methodology
1. Environment Setup
Create a virtual environment and install dependencies:

bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
2. Data Acquisition and Preprocessing
Retrieve and prepare the dataset using:

bash
python scripts/download_dataset.py
python src/preprocessing/preprocess_data.py
This step performs normalization, data augmentation, and partitioning into training, validation, and test subsets.

3. Model Training and Evaluation
Train and evaluate the InceptionV3 model using:

bash
python src/training/train_model.py --epochs 50 --batch_size 32
python src/evaluation/evaluate.py
The model employs transfer learning based on ImageNet weights, retraining the top dense layers for facial emotion classification.

System Architecture
Data Preprocessing: Normalization and augmentation based on FER‑2013 architecture.

Model Training: Fine-tuned InceptionV3 for emotion recognition.

Real-Time Emotion Mapping: Displays parental emotions as emojis to aid child understanding.

Progress Tracking: Monitors emotional learning across three repeats per session.

Audio Sensitivity Analysis: Records facial emotional reactions to different sound frequencies.

Automated Reporting: Summarizes observations and tracks improvement over time.

Results
Successful emotion recognition achieved in real time with notable accuracy using InceptionV3.

Integrated sound irritability detection and progress tracking demonstrated effective data fusion.

The report generator enables caregivers to visualize performance metrics for structured intervention planning.

Required Packages
Dependencies are listed in requirements.txt, including:

text
numpy  
pandas  
opencv-python  
tensorflow  
keras  
matplotlib  
scikit-learn  
flask
Repository Structure
text
Emotion-Aware-Learning/
│
├── src/               # Model training, evaluation, and logic modules
├── scripts/           # Dataset handling scripts
├── data/              # Raw and processed data (excluded from version control)
├── results/           # Reports and visualization outputs
├── requirements.txt   # Installation dependencies
└── .gitignore         # Excluded files and directories
.gitignore
text
__pycache__/
*.pyc
*.h5
*.pkl
*.zip
data/
models/
results/
.env
References
Rashidan, M. et al. (2021). Technology-Assisted Emotion Recognition for Autism Spectrum Disorder. IEEE Access.

Chien, Y. et al. (2023). Game-Based Social Interaction Platform for Cognitive Assessment of Autism Using Eye Tracking. IEEE Transactions.

Bartl-Pokorny, K. D. et al. (2021). Robot-Based Intervention for Children with Autism Spectrum Disorder. IEEE Access.

Prakash, V. G. et al. (2023). Computer Vision-Based Assessment of Autistic Children. IEEE Access.

Kurian, A. & Tripathi, S. (2025). mAutNet: Multimodal Emotion Recognition in Autistic Children. IEEE Access.

Acknowledgment
This project is developed as part of an initiative to create adaptive and inclusive learning systems. It aims to use artificial intelligence for improving emotional awareness, comfort, and communication for children with autism. The study reflects an interdisciplinary effort at the intersection of assistive technology, affective computing, and developmental research.
