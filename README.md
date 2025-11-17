# 🩺 PCOS Detection Using Federated Learning

### *A Comparative Study of Simple CNN, Deep CNN, and ResNet50 Models*

---

## 📌 Overview

This project implements **Federated Learning (FL)** for **PCOS (Polycystic Ovary Syndrome) detection** using ultrasound images.
Instead of sending sensitive patient data to a central server, each client locally trains a deep learning model and only the **model weights are shared**. The server then aggregates the weights using **FedAvg (Federated Averaging)**.

This approach enhances **privacy**, **security**, and **data confidentiality** in medical imaging pipelines.

The project compares three deep learning models:

* **Simple CNN** (Custom)
* **Deep CNN** (Custom)
* **ResNet50** (Transfer Learning)

---

## 📁 Dataset

The dataset consists of ultrasound images categorized into:

```
dataset/
│── normal/
│── pcos/
```

### 📸 Dataset Example

![image](https://github.com/user-attachments/assets/c4753ffa-f110-4b81-b01a-a7caf90206e3) 

Normal

![image](https://github.com/user-attachments/assets/327ee6a5-c0a3-4891-9379-66720e3266cd) 

PCOS

---

## 🧠 Model Architectures

### **1️⃣ Simple CNN (Custom)**

A lightweight 3-layer convolutional neural network designed for baseline performance.

### **2️⃣ Deep CNN (Custom)**

A deeper version with more convolution and pooling layers, improving feature extraction ability.

### **3️⃣ ResNet50**

A powerful pretrained model with residual connections that prevent vanishing gradients and significantly boost accuracy.

---

## 🏗️ Federated Learning Setup

* **Aggregation Method:** FedAvg
* **Number of Clients:** 10
* **Communication Rounds:** 20
* **Local Epochs Per Client:** 2
* **Batch Size:** 128
* **Learning Rate:** 1e-4

Each client receives global weights → trains locally → sends updated weights → server averages them.

---

## 📊 Results

The final evaluation metrics after federated aggregation:

| Model              | Accuracy (%) | Loss (%) |
| ------------------ | ------------ | -------- |
| FL with Simple CNN | **51.81**    | 68.58    |
| FL with Deep CNN   | **72.54**    | 69.00    |
| FL with ResNet50   | **99.22**    | 10.71    |

### 🏆 Highlight

**ResNet50 achieved the highest accuracy of 99.22%**, demonstrating the strength of transfer learning in medical image classification.

---

## 🚀 Key Features

* Uses **Federated Learning** to maintain privacy.
* Compares **3 deep learning architectures**.
* Includes **data preprocessing, augmentation, and client partitioning**.
* Distributed training with **accurate FedAvg implementation**.
* Supports **custom CNNs and pretrained networks**.

---

## 🔮 Future Improvements

* Apply **Secure Aggregation** for enhanced privacy.
* Add **adversarial robustness** to prevent model manipulation.
* Implement **FedProx** or **FedOpt** for faster convergence.
* Deploy models using **FL frameworks** such as Flower or TensorFlow Federated.
