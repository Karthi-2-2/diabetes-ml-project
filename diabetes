import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("/content/diabetes.csv")
df.head()
df.info()
df.describe()
df['Outcome'].value_counts()

# Load your dataset
# df = pd.read_csv('diabetes.csv')

# 1. Visualization for Glucose (Histogram + KDE)
sns.histplot(df['Glucose'], kde=True, color='skyblue')
plt.title('Distribution of Glucose')
plt.xlabel('Glucose Level')
plt.ylabel('Frequency')
plt.savefig('glucose_distribution.png')
plt.show()
plt.clf()

# 2. Visualization for BMI (Histogram + KDE)
sns.histplot(df['BMI'], kde=True, color='olive')
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.savefig('bmi_distribution.png')
plt.show()
plt.clf()

# 3. Count plot for Outcome
sns.countplot(data=df, x='Outcome', palette='viridis')
plt.title('Count Plot of Outcome')
plt.xlabel('Outcome (0: Negative, 1: Positive)')
plt.ylabel('Count')
plt.savefig('outcome_count.png')
plt.show()
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['Pregnancies'] = df['Pregnancies'].replace(0, df['Pregnancies'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
df.describe()
#Prepare Data for Modeling and Scaling
X=df.drop("Outcome", axis=1)
y=df["Outcome"]
rom sklearn.model_selection import train_test_split

#Prepare Data for Modeling and Scaling
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#Feature Scaling (Standardization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Train the Classification Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

#Make Predictions
y_pred = model.predict(X_test_scaled)

#Evaluate Performance
#Confusion Matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
