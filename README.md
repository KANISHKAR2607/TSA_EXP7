### DEVELOPED BY: KANISHKAR M
### REGISTER NO: 212222240044
### DATE:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL

## AIM:
To Implement an Auto Regressive Model for Water Quality data using Python
## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model 
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
## PROGRAM
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/waterquality.csv', parse_dates=['Date'], index_col='Date')

print(data.head())

data= data[np.isfinite(data['pH'])].dropna()

result = adfuller(data['pH'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

model = AutoReg(train['pH'], lags=13)
model_fit = model.fit()

predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test['pH'], predictions)
print('Mean Squared Error:', mse)

plt.figure(figsize=(10,6))
plt.subplot(211)
plot_pacf(train['pH'], lags=13, ax=plt.gca())
plt.title("PACF - Partial Autocorrelation Function")
plt.subplot(212)
plot_acf(train['pH'], lags=13, ax=plt.gca())
plt.title("ACF - Autocorrelation Function")
plt.tight_layout()
plt.show()

print("PREDICTION:")
print(predictions)

plt.figure(figsize=(10,6))
plt.plot(test.index, test['pH'], label='Actual pH value')
plt.plot(test.index, predictions, color='red', label='Predicted pH value')
plt.title('Test Data vs Predictions (FINAL PREDICTION)')
plt.xlabel('Date')
plt.ylabel('pH value')
plt.legend()
plt.show()

```
## OUTPUT:

### GIVEN DATA
![image](https://github.com/user-attachments/assets/761c40fd-a2c9-4092-87e1-5e383e9ad645)

### ADF-STATISTIC AND P-VALUE
![image](https://github.com/user-attachments/assets/ae4f8a29-32fb-466d-8f5f-2c67514b729b)


### PACF - ACF
![image](https://github.com/user-attachments/assets/d4809df8-52f2-450c-b631-33d9b4a7cd8c)

### MSE VALUE
![image](https://github.com/user-attachments/assets/d66b1240-a3f6-48d1-b6c8-41ee64365af8)



### PREDICTION
![image](https://github.com/user-attachments/assets/daff71a4-f64a-4d6f-9370-e286b94e2d23)

### FINAL PREDICTION
![image](https://github.com/user-attachments/assets/1c8ca649-1c98-4490-9750-d70eb2acf9ae)


### RESULT:
Thus, the program to implement the auto regression function using python is executed successfully.
