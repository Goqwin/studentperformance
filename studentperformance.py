import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load the student data
file_path = 'student-mat.csv'
student_data = pd.read_csv(file_path, delimiter=";")

# Display basic information about the DataFrame
print("DataFrame Information:")
print(f"Length: {len(student_data)}")
print(f"Shape: {student_data.shape}")
print(f"Columns: {student_data.columns}")
print(student_data.info())

# Data preprocessing and feature selection
selected_columns = ['goout', 'Dalc', 'Walc', 'G1', 'G2', 'G3']
selected_data = student_data[selected_columns]

# Calculate the correlation matrix
correlation_matrix = selected_data.corr()

# Display correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Select only numeric columns from the dataframe
numeric_columns = ['Dalc', 'goout', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
student_data = student_data[numeric_columns]

# Convert columns to appropriate data types
student_data = student_data.apply(pd.to_numeric, errors='coerce')

# Fill missing values with column means
student_data.fillna(student_data.mean(), inplace=True)

# Feature scaling
scaler = StandardScaler()
X = student_data.drop('G3', axis=1)
y = student_data['G3']
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ridge Regression model
ridge_model = Ridge(alpha=1.0)  
ridge_model.fit(X_train, y_train)

# Evaluation the model
y_pred = ridge_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

# VISUAL ANALYSIS (EDA)
# Scatter plots
plt.figure(figsize=(12, 6))

# Scatter plot of Daily Alcohol Consumption vs Predicted G3 grades
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted')
plt.xlabel('Daily Alcohol Consumption')
plt.ylabel('Predicted G3 Grade')
plt.title('Daily Alcohol Consumption vs Predicted G3 Grade')
plt.legend()

# Scatter plot of Going Out vs Predicted G3 grades
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 1], y_test, color='blue', label='Actual')
plt.scatter(X_test[:, 1], y_pred, color='red', label='Predicted')
plt.xlabel('Going Out')
plt.ylabel('Predicted G3 Grade')
plt.title('Going Out vs Predicted G3 Grade')
plt.legend()

plt.tight_layout()
plt.show()
