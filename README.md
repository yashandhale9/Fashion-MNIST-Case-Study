# 👕 Fashion-MNIST Case Study – Image Classification using CNN & ML Models


---

## 📘 Project Overview

This project focuses on **image classification** using the **Fashion-MNIST dataset** — a collection of 70,000 grayscale images of clothing items.  
It demonstrates both **Deep Learning (CNN)** and **Classical Machine Learning** approaches to identify fashion products such as shirts, shoes, trousers, and bags.

The main goal is to:
- Build and train a **Convolutional Neural Network (CNN)** to classify clothing images.
- Compare its performance with **traditional ML models** like Logistic Regression, SVM, and Random Forest.
- Visualize and save predictions, model checkpoints, and evaluation results.

---

## 🧠 Dataset Details

**Dataset Used:** [Fashion-MNIST (Built-in Keras Dataset)](https://github.com/zalandoresearch/fashion-mnist)  
**Source:** Zalando Research  
**Loaded From:** `tensorflow.keras.datasets.fashion_mnist`

| Property | Description |
|-----------|--------------|
| **Total Images** | 70,000 |
| **Training Set** | 60,000 images |
| **Test Set** | 10,000 images |
| **Image Size** | 28 × 28 pixels (grayscale) |
| **Classes (10)** | T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot |

> 🧩 The dataset is automatically downloaded the first time you run the script. No manual download required.

---

## 🏗️ Project Structure

fashion-mnist-classifier/

├── fashion_mnist_case_study.py # Main project script

├── README.md # Project documentation

└── artifacts/ # Folder for saved models and outputs

├── best_model.keras # Best-performing CNN model

├── final_model.keras # Final saved CNN model

└── inference_grid.png # Visualization of predictions


---

## ⚙️ Features & Functionality

### 1️⃣ Data Loading & Preprocessing
- Loads the Fashion-MNIST dataset using TensorFlow.
- Normalizes pixel values to [0,1].
- Splits the dataset into **Train / Validation / Test** sets.

### 2️⃣ CNN Model Building
- Data augmentation (rotation, zoom, translation).
- Multiple convolutional layers with BatchNorm & Dropout.
- Adam optimizer and Sparse Categorical Crossentropy loss.
- Configurable learning rate and batch size.

### 3️⃣ Model Training
- Includes advanced callbacks:
  - **ModelCheckpoint:** Save best model.
  - **EarlyStopping:** Stop when validation loss stops improving.
  - **ReduceLROnPlateau:** Dynamically adjust learning rate.

### 4️⃣ Baseline Machine Learning Models
- **Logistic Regression**
- **Linear SVM (Support Vector Classifier)**
- **Random Forest Classifier**

### 5️⃣ Inference & Visualization
- Loads the saved best model.
- Makes predictions on random test samples.
- Generates a grid showing:
  - Predicted Label (P)
  - True Label (T)
- Saves the visualization to `artifacts/inference_grid.png`.

---

## 🧩 Installation & Setup

### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/yashandhale9/fashion-mnist-case-study.git
cd fashion-mnist-case-study
```

### **🔹Step 2: Install Required Dependencies**

pip install tensorflow scikit-learn matplotlib numpy


### **🔹 Step 3: Run the Project**

🏋️ Train the CNN Model
python fashion_mnist_case_study.py --train

---

### **🧠 Run Inference on Test Samples**

python fashion_mnist_case_study.py --infer

---

### **⚖️ Run Baseline ML Models**

python fashion_mnist_case_study.py --baselines

---

### **📊 Results & Observations**

| Model               | Type           | Accuracy (Approx.) |
| ------------------- | -------------- | ------------------ |
| Logistic Regression | Classical ML   | ~84%               |
| Linear SVM          | Classical ML   | ~86%               |
| Random Forest       | Classical ML   | ~87%               |
| CNN (Deep Learning) | Neural Network | **~90–92%**        |

---

### **🌟 Topics Covered**

| Concept                       | Description                                    |
| ----------------------------- | ---------------------------------------------- |
| Data Preprocessing            | Normalization, reshaping, validation splitting |
| Convolutional Neural Networks | Conv2D, MaxPooling, Dropout, Dense layers      |
| Model Regularization          | Batch Normalization, Dropout, EarlyStopping    |
| Evaluation & Metrics          | Accuracy, Validation Loss, Test Performance    |
| Machine Learning Comparison   | Logistic Regression, SVM, Random Forest        |
| Model Persistence             | Saving/loading models and visual results       |


---

### **🚀 Future Scope**

Hyperparameter Tuning – Optimize learning rate, batch size, etc.

Transfer Learning – Use pre-trained models like ResNet or MobileNet.

Deployment – Build a web app (Flask/Streamlit) to upload fashion images for predictions.

Visualization Enhancements – Add confusion matrix and training curve plots.

Real-world Dataset – Extend to real fashion product images for e-commerce use.

---

### **⚠️ Limitations**

| Limitation            | Description                                                                    |
| --------------------- | ------------------------------------------------------------------------------ |
| Small Dataset         | Works well on Fashion-MNIST but may not generalize to complex real-world data. |
| Training Time         | CNN requires GPU for faster training.                                          |
| Model Explainability  | CNNs act as black boxes; feature importance isn’t visible.                     |
| Visualization Limited | Confusion matrix and metrics visualization can be added for deeper insights.   |


---

### **🧾 Summary**

| Aspect             | Details                                                     |
| ------------------ | ----------------------------------------------------------- |
| **Goal**           | Classify fashion items using ML & DL models                 |
| **Dataset**        | Fashion-MNIST (Keras built-in)                              |
| **Core Model**     | Convolutional Neural Network                                |
| **Libraries Used** | TensorFlow, Keras, Scikit-learn, Matplotlib, NumPy          |
| **Output**         | Model files, inference grid, performance results            |
| **Use Cases**      | Fashion recognition, e-commerce automation, product tagging |

---

### **🤝 Contributing**

Contributions are always welcome!

If you’d like to improve this project, follow these steps:

Fork the repository

Create a new branch (feature/new-idea)

Commit your changes

Submit a Pull Request 🚀


---

### **🧑‍💻 Author**
👤 Yash Gorakshnath Andhale
