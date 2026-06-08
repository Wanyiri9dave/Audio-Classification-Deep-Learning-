# 🔊 Urban Sound Classification with Deep Learning

A production-ready deep learning pipeline that transforms raw audio waveforms into Log-Mel Spectrograms to classify urban environmental noises using a Convolutional Neural Network (CNN).

---

## 📌 Project Overview

* **Objective:** Automatically identify and classify short urban audio clips into 10 distinct environmental categories (e.g., sirens, street music, drilling).
* **The Challenge:** Raw audio data is irregular in duration and complex to interpret directly. This project leverages computer vision techniques by translating audio frequencies over time into a spatial image format that a neural network can easily learn from.
* **Core Stack:** Python, PyTorch, Librosa, Matplotlib, Pandas, Streamlit.

---

## 📊 The Dataset

This project utilizes the **UrbanSound8K** dataset, which contains 8,732 labeled sound excerpts ($\le$ 4 seconds each). The data is distributed across the following 10 classes:

* Air Conditioner
* Car Horn
* Children Playing
* Dog Bark
* Drilling
* Engine Idling
* Gun Shot
* Jackhammer
* Siren
* Street Music

The dataset is pre-sorted into 10 structural folds to facilitate strict stratified cross-validation and prevent data leakage.

---

## 🛠️ Data Engineering & Feature Pipeline

The core technical strength of this project is the audio preprocessing pipeline. Raw signal data is passed through a multi-stage transformation:

1. **Resampling & Normalization:** All files are downsampled to a uniform 22,050 Hz to balance audio fidelity with computational efficiency.
2. **Short-Time Fourier Transform (STFT):** Raw waveforms are broken down into time-frequency frames.
3. **Log-Mel Conversion:** Frequencies are mapped to the human-centric Mel scale and converted to decibels (`power_to_db`), yielding a 2D matrix (image) of shape `(128, fixed_time_steps)`.
4. **Padding & Trimming:** Clips shorter than 4 seconds are zero-padded, and longer clips are trimmed to ensure every input matrix matches exact dimensions before entering the network.

### Visualizing the Features
Below is an example of a raw audio sample transformed into a Log-Mel Spectrogram. This 2D representation allows our vision model to detect frequency patterns over time.

![Log-Mel Spectrogram](images/spectrogram_sample.png)

---

## 🧠 Model Architecture & Training

I evaluated two distinct approaches to tackle the classification task:

### 1. Custom CNN Baseline
A lightweight, 4-layer Convolutional Neural Network built from scratch. It uses alternating `Conv2D`, `BatchNorm2D`, `ReLU`, and `MaxPooling2D` blocks, capped with a `Dropout(0.4)` layer to minimize overfitting on smaller folds.

### 2. Transfer Learning (ResNet-18)
To optimize performance, I adapted a pre-trained **ResNet-18** architecture. The input layer was modified to accept our single-channel spectrograms, and the final classification head was replaced with a Dense layer outputting 10 logits.

### Cross-Validation Strategy
To prevent data leakage, evaluation was performed using **Stratified 10-Fold Cross-Validation** matching the dataset's native fold design. This ensures chunks of the same original source recording never appear in both the training and testing sets simultaneously.

---

## 📈 Performance & Results

* **Baseline CNN Accuracy:** ~XX.X%
* **Transfer Learning (ResNet-18) Accuracy:** **~XX.X%**

### Evaluation Insights
Below is the final evaluation confusion matrix. 

![Confusion Matrix](images/confusion_matrix.png)

**Key Takeaways:**
* The model achieved near-perfect accuracy on distinct, continuous sounds like *Sirens* and *Jackhammers*.
* Minor confusion occurred between *Dog Barking* and *Children Playing* due to overlapping transient frequency spikes.

---

## 🚀 Interactive Web UI

This project includes an interactive web dashboard built with **Streamlit** that allows users to upload their own `.wav` files and see live inference results alongside the generated spectrogram.

### Running the App Locally

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)[Your-Username]/urban-sound-classifier.git
cd urban-sound-classifier

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit interface
streamlit run app.py
