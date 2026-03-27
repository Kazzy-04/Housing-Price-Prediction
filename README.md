# 🏠 Multimodal Housing Price Prediction

## 📌 Overview

This project demonstrates a **multimodal deep learning approach** for predicting housing prices using:

* **Tabular data** (e.g., square footage, bedrooms, bathrooms, age)
* **Image data** (house images of 64x64 RGB)

The model fuses features from both modalities using a **CNN for images** and a **dense network for tabular data**, combining them for a regression output to predict house prices.

---

## ⚙️ Features

✔ Multimodal input handling (images + tabular data)
✔ CNN-based image feature extraction
✔ Dense layers for tabular data
✔ Feature fusion with concatenation
✔ Regression output for price prediction
✔ Model evaluation with MAE and RMSE
✔ Visualization of training history and predictions

---

## 🛠️ Tech Stack

* **Python**
* **NumPy** & **Pandas**
* **Scikit-learn** (for preprocessing & metrics)
* **TensorFlow / Keras**
* **Matplotlib**

---

## 🔄 Workflow

### 1️⃣ Data Generation & Preprocessing

* Generated **dummy tabular data** with 4 features
* Standardized tabular features using `StandardScaler`
* Generated **dummy image data** (64x64 RGB)
* Created target variable (house prices) with noise
* Split data into training (80%) and testing (20%)

---

### 2️⃣ Model Architecture

**Branch 1 – CNN for Image Features:**

* Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense

**Branch 2 – Dense Network for Tabular Data:**

* Dense → Dense

**Fusion Layer:**

* Concatenate image and tabular features

**Output Layer:**

* Dense → Dense → Dense(1) for regression

**Compilation:**

```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

---

### 3️⃣ Model Training

* Input: `{ "image_input": X_img_train, "tabular_input": X_tab_train }`
* Target: `y_train`
* Validation split: 20%
* Epochs: 10
* Batch size: 32

---

### 4️⃣ Model Evaluation

Metrics used:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**

Sample predictions:

```
House 1: Predicted = $123.45k | Actual = $120.00k | Diff = $3.45k
House 2: ...
```

---

### 5️⃣ Visualizations

1. **Training & Validation Loss**

   * Monitors convergence of model training

2. **Actual vs Predicted Prices**

   * Scatter plot comparing predicted and actual housing prices
   * Red dashed line indicates perfect predictions

---

## 🚀 Getting Started

### 🔧 Installation

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

---

### ▶️ Running the Project

```bash
python Task-3(Housing Prediction).py
```

---

### 📁 Dataset

This project uses **synthetic/dummy data** for demonstration purposes. For real applications, replace with actual housing datasets and corresponding images.

---

## 📌 Future Improvements

* Use real-world datasets (e.g., Zillow, Kaggle housing datasets)
* Experiment with **pretrained CNNs** for image features (e.g., ResNet, EfficientNet)
* Hyperparameter tuning for better regression performance
* Deploy as a web app for interactive predictions


