import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import math

# Load dataset
df = pd.read_csv("customer_churn.csv")

# --- 1. Data Manipulation ---

# Extract the 5th column
customer_5 = df.iloc[:, 4]
# Extract the 15th column
customer_15 = df.iloc[:, 14]

# Extract all male senior citizens with electronic check payment
senior_male_electronic = df[(df['gender']=='Male') & 
                             (df['SeniorCitizen']==1) &
                             (df['PaymentMethod']=='Electronic check')]

# tenure > 70 OR MonthlyCharges > 100
customer_total_tenure = df[(df['tenure']>70) | (df['MonthlyCharges']>100)]

# Two year contract + mailed check + Churn = Yes
two_mail_yes = df[(df['Contract']=='Two year') & 
                  (df['PaymentMethod']=='Mailed check') & 
                  (df['Churn']=='Yes')]

# 333 random records
customer_333 = df.sample(n=333, random_state=42)

# Count of churn
churn_count = df['Churn'].value_counts()
print("Churn Counts:\n", churn_count)

# --- 2. Data Visualization ---

# Bar plot for InternetService
plt.figure(figsize=(6,4))
df['InternetService'].value_counts().plot(kind='bar', color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of Categories')
plt.title('Distribution of Internet Service')
plt.show()

# Histogram for tenure
plt.figure(figsize=(6,4))
plt.hist(df['tenure'], bins=30, color='green')
plt.title('Distribution of tenure')
plt.xlabel('Tenure')
plt.ylabel('Count')
plt.show()

# Scatter plot between MonthlyCharges and tenure
plt.scatter(df['tenure'], df['MonthlyCharges'], color='brown')
plt.xlabel('Tenure of customer')
plt.ylabel('Monthly Charges of customer')
plt.title('Tenure vs Monthly Charges')
plt.show()

# Box plot between tenure and Contract
plt.figure(figsize=(6,4))
sns.boxplot(x='Contract', y='tenure', data=df)
plt.title('Contract vs Tenure')
plt.show()

# --- 3. Linear Regression ---
X = df[['tenure']]
y = df['MonthlyCharges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

error = y_test - y_pred
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print("Linear Regression RMSE:", rmse)

# --- 4. Logistic Regression ---
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# Single logistic regression (MonthlyCharges)
X = df[['MonthlyCharges']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Logistic Regression (1 variable) Accuracy:", accuracy_score(y_test, y_pred))

# Multiple logistic regression (tenure, MonthlyCharges)
X = df[['tenure','MonthlyCharges']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

log_reg2 = LogisticRegression(max_iter=200)
log_reg2.fit(X_train, y_train)
y_pred = log_reg2.predict(X_test)
print("Multiple Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# --- 5. Decision Tree ---
X_train, X_test, y_train, y_test = train_test_split(df[['tenure']], y, test_size=0.20, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

# --- 6. Random Forest ---
X_train, X_test, y_train, y_test = train_test_split(df[['tenure','MonthlyCharges']], y, test_size=0.30, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
