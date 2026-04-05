# 🫁Interpretable-Chest-XRay-based-Pneumonia-and-Covid-19-detection-using-CNN-and-GradCam

A deep learning-based web application that detects **Pneumonia** and **COVID-19** from chest X-ray images using a Convolutional Neural Network (CNN), deployed via a Flask web application.

---

## 📌 Overview

Early and accurate detection of respiratory diseases like Pneumonia and COVID-19 is critical for timely treatment. This project leverages the power of deep learning to classify chest X-ray images into three categories:

- **Normal** – Healthy lungs
- **Pneumonia** – Bacterial or viral pneumonia
- **COVID-19** – COVID-19 infected lungs

The trained model achieves a classification accuracy of **~92%** and is served through an interactive Flask web application.

---

## 🗂️ Project Structure

```
Pneumonia-and-Covid-19-detection-using-Chest-XRAY/
│
├── static/
│   └── images/             # Static image assets for the web app
│
├── templates/              # HTML templates for the Flask app
│
├── app.py                  # Flask application entry point
├── main.py                 # Core prediction/inference logic
├── train.ipynb             # Jupyter notebook for model training
├── chest_xray.h5           # Pre-trained Keras model weights
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python runtime version (for deployment)
├── Procfile                # Process configuration (for Heroku deployment)
└── README.md
```

---

## 🧠 Model Architecture

The model is a **Convolutional Neural Network (CNN)** trained on chest X-ray images. Key steps:

| Stage               | Details                                          |
|---------------------|--------------------------------------------------|
| **Input**           | Chest X-ray images (grayscale/RGB)               |
| **Classes**         | Normal, Pneumonia, COVID-19                      |
| **Framework**       | TensorFlow / Keras                               |
| **Model file**      | `chest_xray.h5`                                  |
| **Accuracy**        | ~92%                                             |
| **Deployment**      | Flask web app                                    |

---

## 📊 Dataset

The dataset consists of labeled chest X-ray images across **3 classes**:

- **Normal** – Healthy chest X-rays
- **Pneumonia** – X-rays showing pneumonia infection
- **COVID-19** – X-rays showing COVID-19 infection

> **Note:** Data is preprocessed (resizing, normalization, augmentation) and loaded via a data loader before training.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.x
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/JananiVenk/Pneumonia-and-Covid-19-detection-using-Chest-XRAY.git
cd Pneumonia-and-Covid-19-detection-using-Chest-XRAY
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask App

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`.

---

## 🖥️ Usage

1. Launch the Flask web application.
2. Upload a chest X-ray image through the web interface.
3. The model processes the image and returns one of the following predictions:
   - ✅ **Normal**
   - ⚠️ **Pneumonia**
   - 🔴 **COVID-19**

---

## 🏋️ Training the Model

To retrain the model from scratch, open and run the Jupyter notebook:

```bash
jupyter notebook train.ipynb
```

The notebook covers:
- Data loading and preprocessing
- Data augmentation
- CNN model definition
- Model training and validation
- Saving the trained model as `chest_xray.h5`

---

## 🚀 Deployment

This project is configured for deployment on **Heroku** using:

- `Procfile` – Defines the web process (`web: gunicorn app:app`)
- `runtime.txt` – Specifies the Python version
- `requirements.txt` – Lists all dependencies

To deploy:

```bash
heroku create
git push heroku main
```

---

## 📦 Dependencies

Key libraries used (see `requirements.txt` for full list):

| Library         | Purpose                        |
|-----------------|--------------------------------|
| TensorFlow/Keras| Model building & inference     |
| Flask           | Web application framework      |
| NumPy           | Numerical operations           |
| Pillow          | Image processing               |
| Gunicorn        | WSGI server for deployment     |

---

## 📈 Results

| Class       | Description                       |
|-------------|-----------------------------------|
| Normal      | No infection detected             |
| Pneumonia   | Bacterial/viral pneumonia present |
| COVID-19    | COVID-19 infection detected       |

**Overall Model Accuracy: ~92%**

---

## 📝 License

This project is open-source. Feel free to use, modify, and distribute with attribution.

---

## 🙋‍♀️ Author

**JananiVenk**  
GitHub: [@JananiVenk](https://github.com/JananiVenk)

---

> ⚠️ **Disclaimer:** This tool is intended for educational and research purposes only. It should **not** be used as a substitute for professional medical diagnosis.
