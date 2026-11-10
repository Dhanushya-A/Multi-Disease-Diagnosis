# Multi-Disease Diagnosis from Medical Images Using Machine Learning and Deep Learning

## Project Overview

The **Multi-Disease Diagnosis System** is a medical image analysis project that explores Machine Learning (ML) and Deep Learning (DL) techniques to identify multiple diseases from medical images.

The project focuses on processing medical images, extracting meaningful features, training deep learning models, and predicting possible disease categories.

It is designed as an AI-assisted research and educational project.

## Objectives

- Analyze medical images using image processing techniques.
- Develop deep learning models for disease classification.
- Explore CNN-based architectures such as ResNet and DenseNet.
- Apply machine learning techniques for classification and prediction.
- Evaluate model performance using appropriate evaluation metrics.
- Provide a simple interface for image-based predictions.

## Technologies Used

| Category             | Technologies         |
| -------------------- | -------------------- |
| Programming Language | Python               |
| Deep Learning        | TensorFlow / Keras   |
| Machine Learning     | Scikit-learn         |
| Image Processing     | OpenCV, Pillow       |
| Data Analysis        | Pandas, NumPy        |
| Visualization        | Matplotlib           |
| Web Application      | Flask                |
| Development Tools    | VS Code, Git, GitHub |

## Project Structure

```text
Multi-Disease-Diagnosis/
│
├── app/
│   ├── app.py
│   └── utils.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── create_dataset.py
│   ├── data_generator.py
│   ├── dataset_analysis.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── reduce_dataset.py
│   └── train.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Project Workflow

1. **Dataset Collection:** Obtain the required medical image dataset.
2. **Dataset Analysis:** Analyze image distribution and dataset information.
3. **Preprocessing:** Resize images, normalize pixel values, and prepare the data.
4. **Dataset Preparation:** Organize the dataset for training, validation, and testing.
5. **Model Development:** Build deep learning models for image classification.
6. **Model Training:** Train the selected model using prepared medical images.
7. **Evaluation:** Evaluate the model using classification metrics.
8. **Prediction:** Use the trained model to generate predictions for new images.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Dhanushya-A/Multi-Disease-Diagnosis.git
```

### 2. Navigate to the project directory

```bash
cd Multi-Disease-Diagnosis
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset Information

The project uses medical image data for disease classification and analysis.

The dataset is not included in this GitHub repository because of its size. Download and organize the dataset separately before running the preprocessing and training scripts.

Place the dataset in the appropriate directories according to the configuration in `src/config.py`.

## Running the Project

Run the scripts from the project root directory in the following general order:

### 1. Dataset analysis

```bash
python src/dataset_analysis.py
```

### 2. Dataset preparation

```bash
python src/create_dataset.py
```

### 3. Image preprocessing

```bash
python src/preprocessing.py
```

### 4. Model training

```bash
python src/train.py
```

### 5. Model evaluation

```bash
python src/evaluate.py
```

### 6. Run the web application

```bash
python app/app.py
```

**Note:** Follow the dataset paths and configuration settings in the source files. Some scripts may require the dataset or a trained model to be prepared first.

## Evaluation Metrics

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Actual evaluation results should be added after completing model training and testing.

## Future Enhancements

- Improve classification performance using advanced deep learning architectures.
- Support additional medical image types and disease categories.
- Add Grad-CAM-based visual explanations.
- Improve the web interface and prediction experience.
- Explore model optimization and deployment.

## Disclaimer

This project is developed for educational and research purposes only. Its predictions are not medical diagnoses and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Author

**Dhanushya A**

B.Sc. Information Technology Graduate

- GitHub: [https://github.com/Dhanushya-A](https://github.com/Dhanushya-A)
- Project Repository: [https://github.com/Dhanushya-A/Multi-Disease-Diagnosis](https://github.com/Dhanushya-A/Multi-Disease-Diagnosis)
