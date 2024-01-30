import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score


#provide title to the web app
st.title("Multiple Linear Regression")

# Load the dataset
df = pd.read_excel("Dropoutdataset.xlsx")


# Display dataset info
st.write("### Dataset Info:")
# Display dataset info
st.write("### Dataset Info:")
st.write("Number of Rows:", df.shape[0])
st.write("Number of Columns:", df.shape[1])
st.write("Column Names:", df.columns.tolist())
st.write("Data Types:", df.dtypes)
st.write("Number of Missing Values:", df.isnull().sum())


# Display missing values
st.write("### Missing Values:")
missing_values = df.isnull().sum()
st.write(missing_values)

# Encode target column using LabelEncoder
university = LabelEncoder()
df['Target'] = university.fit_transform(df['Target'])

# Display updated DataFrame
st.write("### Updated DataFrame:")
st.write(df.head())

# Use OneHotEncoder to encode categorical features
ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), ['Target'])], remainder="passthrough")
X = df.iloc[:, 0:34]
y = df.iloc[:, 34]
y_encoded = ct.fit_transform(df[['Target']])

# Display the transformed data
st.write("### Transformed Data:")
st.write(pd.DataFrame(y_encoded).head())

# Split the data into training and testing sets
x_train, X_test, y_train, Y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

# Fit multiple linear regression to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting results
y_pred = regressor.predict(X_test)

# Calculating coefficients and intercepts
coefficients = regressor.coef_
intercept = regressor.intercept_

# Calculating the R squared value
r2_value = r2_score(Y_test, y_pred)

# Display results
st.write("### Results:")
st.write(f"Coefficients: {coefficients}")
st.write(f"Intercept: {intercept}")
st.write(f"R-squared Value: {r2_value}")








# button for Histogram
if st.button("Generate Histogram Chart"):
    #plot a bar chart
    selected_columns = st.multiselect("Select the columns to visualise the Histogram chart",df.columns)
    if selected_columns:
        st.bar_chart(df[selected_columns])
    else:
        st.warning("please select atleast two columns.")
        
        
        
        
# User input for independent variables
st.sidebar.title ("Enter values to be Predicted")

# create the input for each feature

user_input = {}
for feature in df.columns[:-1]:
    user_input[feature] = st.sidebar.text_input(f"Enter {feature}",0.0)
    

# Button to triger the prediction

if st.sidebar.button("Predict"):
    
    #create a dataframe for the user input
    user_input_df = pd.DataFrame([user_input],dtype= float)
    
    #predict using the trained model
    y_pred = regressor.predict(user_input_df)
    
    #inverse transform to get the original target values
    
    predicted_class = university.inverse_transform(np.array(y_pred, axis=1))
    
    #display the predicted class/target
    
    st.write('### Predicted Class')
    st.write(predicted_class[0])
      

#------------------------------------------------------------------------------------------------------------------------------------

# Button to trigger bar chart
if st.button("Generate Bar Chart"):
    selected_columns = st.multiselect("Select columns:", df.columns)
    if selected_columns:
        st.altair_chart(df["Inflation rate"],df["Target"])
    else:
        st.warning("Please select at least two columns.")

# Button for line chart
if st.button("Generate Line Chart"):
    selected_columns = st.multiselect("Select columns:", df.columns)
    if selected_columns:
        st.line_chart(df[selected_columns])
    else:
        st.warning("Please select at least two columns.")

# Feature engineering
university = LabelEncoder()
df['Target'] = university.fit_transform(df['Target'])

ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), ['Target'])], remainder='passthrough')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y_encoded = ct.fit_transform(df[["Target"]])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

# Fit the regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# User input for prediction
st.sidebar.title("Enter values to be Predicted")

user_input = {}
for feature in df.columns[:-1]:
    user_input[feature] = st.sidebar.text_input(f"Enter {feature}", 0.0)

# Prediction button
if st.sidebar.button("Predict"):
    user_input_df = pd.DataFrame([user_input])
    user_input_encoded = ct.transform(user_input_df)  # Transform all features
    y_pred = regressor.predict(user_input_encoded)
    predicted_class = university.inverse_transform(np.array(y_pred, dtype=int).reshape(-1, 1))
    st.write('### Predicted Class')
    st.write(predicted_class[0][0])