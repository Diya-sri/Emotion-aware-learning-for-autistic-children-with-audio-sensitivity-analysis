Emotion-Aware Learning for Autistic Children with Audio Sensitivity Analysis
1. Abstract

This project introduces a web-based application designed to support children with autism by combining real-time emotion recognition and audio sensitivity analysis. The system detects the parent’s emotions through facial expressions, converts them into appropriate emojis, and displays them on the child’s interface to aid emotional understanding. The child’s responses are observed across three attempts, allowing for structured learning and progress tracking. Additionally, parents can upload different audio samples to assess whether certain sounds cause discomfort, helping identify triggers for irritation. A parental report is automatically generated to summarize learning responses and emotional patterns.

2. Project Overview

This repository contains the source code and associated methodological documentation for a web-based application designed to facilitate emotion-aware learning among children with autism spectrum disorder (ASD). The project integrates computer vision-based emotion recognition and audio sensitivity analysis to support structured interventions. The system records, analyzes, and reports interactions in real time, assisting caregivers and researchers in the assessment and support of autistic children.

3. Dataset Description

Dataset: FER-2013 (Facial Expression Recognition)

Download source: https://www.kaggle.com/datasets/msambare/fer2013

Download command:

text
python scripts/download_dataset.py

The dataset includes 35,887 grayscale, 48x48 pixel images of human faces, each categorized into one of seven emotion classes. The data are preprocessed through normalization and augmentation prior to model training.

4. Methodology

4.1 Environment Setup

To run the project, begin by installing required dependencies:

text
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

4.2 Data Acquisition and Preprocessing
Run the following commands to retrieve and prepare the data:

text
python scripts/download_dataset.py
python src/preprocessing/preprocess_data.py

This stage includes normalization and data augmentation.

4.3 Model Training and Evaluation
Train the InceptionV3 model using:

text
python src/training/train_model.py --epochs 50 --batch_size 32
Evaluate performance with:

text
python src/evaluation/evaluate.py

The code employs an InceptionV3 convolutional neural network, adapted via transfer learning to classify facial expressions from the FER-2013 dataset.

5. System Architecture

Data Ingestion and Preprocessing: Handles normalization and augmentation of FER-2013 images.

Model Training: Customizes and fine-tunes InceptionV3 for emotion classification.

Learning Module: Delivers real-time visual cues (emojis) based on parent emotion recognition.

Progress Tracking: Records and structures child responses across multiple attempts.

Audio Sensitivity Analysis: Evaluates emotional responses to various auditory stimuli.

Automated Reporting: Generates and disseminates detailed session reports to caregivers.

6. Results

Demonstrated real-time emotion recognition and emoji mapping.

Multi-attempt tracking and summary reporting functionality implemented.

Experimental validation with the FER-2013 dataset confirmed accuracy in emotion detection under controlled conditions.

7. Required Packages

A full list of dependencies is maintained in requirements.txt. Core libraries include:

numpy

pandas

opencv-python

tensorflow

keras

matplotlib

scikit-learn

flask

8. File Structure

/src/: Source code for model, preprocessing, evaluation.

/scripts/: Helper scripts for dataset management.

/data/: Intended location for raw and processed data (excluded from version control).

/results/: Generated reports and visualizations.

9. .gitignore

All intermediate, output, and sensitive files are excluded from source tracking with the following .gitignore:

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

10. References

Rashidan et al. (2021). Technology-Assisted Emotion Recognition for Autism Spectrum Disorder. IEEE Access.

Chien et al. (2023). Game-Based Social Interaction Platform for Cognitive Assessment of Autism Using Eye Tracking. IEEE Transactions.

Bartl-Pokorny et al. (2021). Robot-Based Intervention for Children With Autism Spectrum Disorder. IEEE Access.

Prakash et al. (2023). Computer Vision-Based Assessment of Autistic Children. IEEE Access.

Kurian & Tripathi (2025). mAutNet: Multimodal Emotion Recognition in Autistic Children. IEEE Access.
