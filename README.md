<div align="center">
  
  # 🐄 PashuPehchaan
  **Advanced Animal Breed Classification & Identification System**
  
  [![Django](https://img.shields.io/badge/Django_4.2-092E20?style=for-the-badge&logo=django&logoColor=white)](https://www.djangoproject.com/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow_2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
  [![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

  *A high-performance Django web application powered by Deep Learning (Keras/TensorFlow) designed to accurately classify animal breeds from images. Get top-3 predictions instantly through an intuitive, mobile-friendly interface.*
</div>

---

## 📑 Table of Contents

- [🚀 Key Features](#-key-features)
- [🛠️ Technology Stack](#️-technology-stack)
- [🏗️ Project Architecture](#️-project-architecture)
- [📦 Setup & Installation](#-setup--installation)
- [🧠 ML Model Integration](#-ml-model-integration)
- [📄 License](#-license)

---

## 🚀 Key Features

📸 **Instant Breed Recognition:** Upload an image of an animal to instantly receive the Top-3 most probable breed classifications.  
🧠 **Deep Learning Core:** Powered by a customized TensorFlow/Keras convolutional neural network (CNN) model.  
⚡ **Real-Time Processing:** Fast backend image preprocessing using Pillow and NumPy to ensure quick inference times.  
🎨 **Intuitive UI/UX:** Clean, responsive front-end designed for accessibility on both desktop and mobile devices.  
🔒 **Secure Uploads:** Safe handling of user-uploaded media with Django's robust static and media file management.  

---

## 🛠️ Technology Stack

### Backend Engine
- **Django (v4.2.14):** Robust, scalable Python web framework.
- **SQLite3:** Lightweight database for handling session and upload data.

### Machine Learning & Processing
- **TensorFlow (v2.15.0):** Core engine for loading the `.h5` model and running inferences.
- **NumPy:** High-performance array operations for image tensor transformations.
- **Pillow (PIL):** Standard Python image processing library for resizing and formatting inputs.

---

## 🏗️ Project Architecture

```text
PashuPehchaan_front_web/
├── breedui/                # Main Django Project Configuration
├── classify/               # Core Application App
│   ├── ml/                 # Directory for Machine Learning Models (.h5 & labels.json)
│   ├── static/             # CSS, JS, and UI Assets
│   ├── templates/          # HTML Templates for the Front-End
│   └── views.py            # Inference Logic & HTTP Handling
├── media/                  # Directory for Temporary Uploaded Images
├── manage.py               # Django CLI utility
└── requirements.txt        # Python Dependencies
```

---

## 📦 Setup & Installation

### 1. Environment Setup
Create and activate a virtual environment to isolate the dependencies:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install all required Python packages (Django, TensorFlow, Pillow, NumPy):
```bash
pip install -r requirements.txt
```

### 3. Database Migration
Initialize the Django SQLite database:
```bash
python manage.py migrate
```

### 4. Run the Development Server
Start the local server to run the application:
```bash
python manage.py runserver
```
> 🎉 Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your web browser.

---

## 🧠 ML Model Integration

For the app to successfully predict breeds, you must provide your trained model file and the corresponding labels map. If you don't have a model yet, the UI will safely show an error message instructing you where to place it.

1. Locate your trained Keras model (`model.h5`) and class labels (`labels.json`).
2. Copy them into the `classify/ml/` directory:
```bash
# Example
cp ../path_to_exports/model.h5 classify/ml/
cp ../path_to_exports/labels.json classify/ml/
```

*Note: The model should be compiled and ready for `.predict()` calls, expecting the input shape defined in your `views.py` preprocessing logic.*

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📄 License
This project is open-source and available for research and educational purposes.
