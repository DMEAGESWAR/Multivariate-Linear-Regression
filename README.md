# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
- Import necessary libraries (e.g., NumPy, Pandas, Scikit-learn)
- Load your dataset into a Pandas DataFrame


### Step2
- Handle missing values (e.g., imputation, removal)
- Scale/normalize features (e.g., StandardScaler)


### Step3
- Split data into training (~70-80%) and testing sets (~20-30%)


### Step4
- Create a Multivariate Linear Regression model (e.g., Scikit-learn's LinearRegression)
 
- Fit the model to the training data

- Evaluate the model using metrics like MSE, R-squared, and MAE


### Step5
- Refine the model by tuning hyperparameters or feature engineering
- Deploy the final model in your desired application or platform

## Program:
```
import pandas as pd

from sklearn import linear_model

df = pd.read_csv("C:\\Users\\admin\\Downloads\\car (1).csv")

X = df[['Weight', 'Volume']]

y = df['CO2']

regr = linear_model.LinearRegression()

regr.fit(X, y)

print('Coefficients:', regr.coef_)

print('Intercept:',regr.intercept_)

predictedCO2 = regr.predict([[3300, 1300]])

print('Predicted CO2 for the corresponding weight and volume',predictedCO2)






```
## Output:

### Insert your output

![Screenshot 2024-12-05 214705](https://github.com/user-attachments/assets/aabb9560-9e4d-467f-8a7c-a25c8caa4b4f)



## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
