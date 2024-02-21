#Recomend to run the Jupyter file for better understanding
#Import the reqired Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Importing the dataset
data = pd.read_csv("data.csv")
data.columns

#Discarding the object type features
data = data[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
       'yr_built', 'yr_renovated']]

#checking for null values
data.dropna(inplace=True)
data.info()

#
data

#printing an histogram for all the features
data.hist(figsize=(20,20))

#printing a correlation heatmap for price in given dataset
plt.figure(figsize=(12,12))
heatmap = sns.heatmap(data.corr(numeric_only=True), vmin=-1, vmax=1, annot=True, cmap ='coolwarm')

#selecting the more relevant features and dividing thenm into dependent and independent features
X, y = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition']], data["price"]

#spliting the data into test and train parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating a Linear Regression model and fitting it.
lm = LinearRegression()
lm.fit(X_train, y_train)

#calculating the mean squared error
y_pred = lm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

#creating aa Actual Price vs Predicted Price graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

#creating aa Residual Price graph
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

#Predicting Price of a house for give features.
new_data = [[3, 2, 1500, 4000, 1, 0, 0, 3]]
predicted_price = lm.predict(new_data)

print("Predicted Price:", predicted_price[0])