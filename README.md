Medical Insurance Cost Prediction Using Machine Learning

This project predicts medical insurance claim costs based on patient demographics, lifestyle attributes, and health factors.
A Linear Regression model is trained using a structured insurance dataset containing medical and demographic data.

✅ Project Objective

To build a machine learning model that accurately estimates medical insurance claim amounts using features like:

Age

Sex

BMI

Weight

Hereditary Diseases

Number of Dependents

Smoking Status

City

Blood Pressure

Diabetes

Exercise Habits

Job Title

📂 Dataset Information

The project uses the provided dataset medicalinsurance.csv with 15,000 records and 13 columns.

Feature	Description
age	Age of the individual
sex	Gender (male/female)
weight	Weight in kg
bmi	Body Mass Index
hereditary_diseases	Genetic/Family disease history
no_of_dependents	Number of dependents
smoker	Whether the person smokes (1/0)
city	City of residence
bloodpressure	Blood pressure reading
diabetes	Diabetes status (1/0)
regular_ex	Regular exercise (1/0)
job_title	Job category
claim	Target Variable - Medical insurance amount (₹)
🧠 Tech Stack

Python

Pandas, NumPy

Scikit-Learn

Matplotlib, Seaborn

Jupyter Notebook

🛠️ Workflow

Import and explore the dataset

Handle missing values (age, bmi)

Perform Exploratory Data Analysis (EDA)

Encode categorical variables (One-Hot Encoding)

Split data into training & testing sets

Train Linear Regression model

Evaluate using MAE, RMSE, R² Score

Save model for deployment

📊 Model Evaluation Metrics
Metric	Meaning
MAE	Mean Absolute Error
RMSE	Root Mean Square Error
R² Score	Goodness of Fit

The results will display after code execution.

📁 Project Files
File	Purpose
medicalinsurance.csv	Dataset
insurance_prediction.ipynb	Notebook with complete workflow
insurance_prediction_model.pkl	Trained ML model
README.md	Project Explanation
▶️ How to Run
# Clone repository
git clone <your-repo-url>

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook insurance_prediction.ipynb

📦 Install Required Libraries
pip install pandas numpy scikit-learn matplotlib seaborn joblib

🎯 Future Enhancements

Include Random Forest / XGBoost models

Build a Streamlit / Flask web interface

Deploy model on cloud (AWS / Heroku / Streamlit Cloud)

Add SHAP / LIME for Explainable AI

🙌 Contributions

Feel free to fork this repo and submit pull requests. Suggestions and improvements are welcome!
