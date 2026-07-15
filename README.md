# 🎤 AI Speaker Recognition System (Deep ResNet-v2)

An end-to-end deep learning system that identifies individuals from 3-second voice clips with high precision. By treating audio classification as a computer vision task, this project achieves professional-grade accuracy in speaker identification.

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

---

## 🌐 Live Demo
**Try the app here:** [https://speaker-recognition-app.streamlit.app/](https://speaker-recognition-app.streamlit.app/)

---

## 🚀 Project Overview
This project solves the challenge of multi-class speaker identification (50 unique speakers). Instead of using traditional, compressed audio features (MFCCs), this model converts raw audio into high-fidelity visual frequency representations (**Log-Mel Spectrograms**) and applies advanced **Residual Network (ResNet)** architectures to classify the unique "voice fingerprints" of each speaker.

## 📊 Key Performance Metrics
| Metric | Performance |
| :--- | :--- |
| **Testing Accuracy** | **92.60%** |
| **Training Accuracy** | **99.16%** |
| **Model Architecture** | ResNet-v2 (Custom Residual CNN) |
| **Hyperparameter Tuning**| Optuna (Bayesian Optimization) |

## 🛠️ Technical Highlights
* **High-Performance Audio Pipeline:** Transitioned from legacy MFCCs to 2D **Log-Mel Spectrograms (128x129)**, resulting in a **40% increase in model accuracy**.
* **Custom ResNet-v2 Architecture:** Implemented a deep Residual CNN with 3 stages of residual blocks, Batch Normalization, and Dropout (0.5) to ensure stable training and robust feature extraction.
* **Bayesian Hyperparameter Tuning:** Leveraged **Optuna** to execute a Bayesian Search, automating the selection of optimal learning rates (0.00055) and weight decay parameters.
* **Robust Data Handling:** Integrated Data Augmentation (Gaussian noise injection and pitch-shifting) to create a model resilient to real-world acoustic variations.

## 💻 Tech Stack
* **Deep Learning:** PyTorch, Librosa (Audio Analysis), NumPy
* **Optimization:** Optuna
* **Deployment/UI:** Streamlit

## 📁 File Structure
* `app.py`: The production-ready inference interface built with Streamlit.
* `model_def.py`: The architecture definition for the custom ResNet-v2 class.
* `requirements.txt`: Project dependencies for deployment.
* `FINAL_speaker_model.pth`: The trained model weights.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Aanchalbs-iitj/speaker-recognition-app.git](https://github.com/Aanchalbs-iitj/speaker-recognition-app.git)
   cd speaker-recognition-app
