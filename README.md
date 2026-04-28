# 🩺 Diabetes Prediction Project

## 📌 Overview
This project predicts whether a person is likely to have diabetes using machine learning techniques.  
It uses the Pima Indians Diabetes dataset and applies data preprocessing, visualization, and classification.

---

## 🎯 Objective
- Analyze diabetes dataset
- Perform data cleaning & preprocessing
- Visualize important features
- Build a prediction model using Logistic Regression

---

## 📂 Project Structure

diabetes-ml-project/
│
├── data/                # Dataset
│   └── diabetes.csv
│
├── images/              # Visualizations
│   ├── glucose_distribution.png
│   ├── bmi_distribution.png
│   └── outcome_count.png
│
├── diabetes.py          # Main Python script
├── requirements.txt     # Libraries used
├── README.md            # Project documentation

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🔍 Data Preprocessing

- Replaced zero values with median (Glucose, BMI, etc.)
- Feature scaling using StandardScaler
- Train-test split (70%-30%)

---

## 📊 Visualizations

### Glucose Distribution
![Glucose](images/glucose_distribution.png)

### BMI Distribution
![BMI](images/bmi_distribution.png)

### Outcome Count
![Outcome](images/outcome_count.png)

---

## 🤖 Model Used

- Logistic Regression

---

## 📈 Results

- Accuracy: ~75% (approx)
- Model performs reasonably well for classification

---
## ⚙️ How to Run

1. Clone the repository:
git clone https://github.com/your-username/diabetes-ml-project.git

2. Navigate to project folder:
cd diabetes-ml-project

3. Install dependencies:
pip install -r requirements.txt

4. Run the script:
python diabetes.py
## 👨‍💻 Author

Karthikeyan
## 🚀 Future Improvements

- Use advanced models (Random Forest, XGBoost)
- Build a web app using Streamlit
- Deploy the model online
