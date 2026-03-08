# Machine Learning Projects Collection

This repository contains two machine learning projects built using Python, Flask, and Deep Learning techniques.

These projects demonstrate practical applications of Machine Learning including Natural Language Processing and Computer Vision.

---

## Projects Included

### 1. Email Spam Classifier

A machine learning model that classifies emails as **Spam** or **Not Spam** using Natural Language Processing techniques.

Features:
- Text preprocessing
- TF-IDF vectorization
- Machine Learning classification
- Flask web interface

Technologies Used:
- Python
- Scikit-learn
- Pandas
- Flask

Project Structure:

email-spam-classifier/
- app.py (Flask web application)
- train_spam_model.py (training script)
- run_spam.py (run trained model)
- spam_model.pkl (trained model)
- vectorizer.pkl (text vectorizer)
- spam.csv (dataset)
- requirements.txt
- templates/

---

### 2. MNIST Digit Recognizer

A deep learning model that recognizes handwritten digits using the famous MNIST dataset.

Features:
- Convolutional Neural Network (CNN)
- Image preprocessing
- Deep learning training
- Flask interface for digit prediction

Technologies Used:
- Python
- TensorFlow / Keras
- NumPy
- Flask

Project Structure:

mnist-digit-recognizer/
- app.py (Flask web application)
- train_mnist_digit.py (model training script)
- run_mnist_digit.py (prediction script)
- mnist_cnn_model.h5 (trained CNN model)
- digit.png (sample image)
- requirements.txt
- templates/

---

## Installation

Clone the repository:

```bash
git clone https://github.com/hatim-khann/ml-projects-collection.git
cd ml-projects-collection
```

---

## Run Email Spam Classifier

```bash
cd email-spam-classifier
pip install -r requirements.txt
python app.py
```

---

## Run MNIST Digit Recognizer

```bash
cd mnist-digit-recognizer
pip install -r requirements.txt
python app.py
```

---
