# 🩺 Pneumonia Detection Web App

A **Streamlit-based web application** for detecting **Pneumonia** from chest X-ray images using a pre-trained deep learning model (TensorFlow/Keras).  
This app allows users to upload an X-ray image, process it, and get a prediction result — whether the patient is likely to have **Pneumonia** or not — along with the model’s confidence score.

---

## 🚀 Features

- 🧠 **Deep Learning Model** trained with TensorFlow/Keras for pneumonia classification  
- 🖼️ **Image Upload & Preview** — supports `.jpg`, `.jpeg`, and `.png` formats  
- 📊 **Prediction Results** — shows the predicted class and confidence percentage  
- 🎨 **Custom UI** built with Streamlit and HTML/CSS for a clean user interface  
- ⚡ **Fast Inference** with TensorFlow SavedModel (via `TFSMLayer`)  
- ☁️ **Deployable on Streamlit Cloud** or any Python web environment  

---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Framework** | Streamlit |
| **Deep Learning** | TensorFlow 2.20.0 + Keras 3 |
| **Image Processing** | Pillow (PIL) |
| **Language** | Python 3.12 |
| **Model Format** | TensorFlow SavedModel (`pneumonia_model_tf/`) |

---

## 🧠 Model Information

The model (`pneumonia_model_tf`) was trained to classify chest X-ray images into:
- **Normal**
- **Pneumonia**

It uses convolutional neural networks (CNNs) trained on a labeled dataset of X-ray images.  
During deployment, it is loaded using the `keras.layers.TFSMLayer` API for efficient inference.
