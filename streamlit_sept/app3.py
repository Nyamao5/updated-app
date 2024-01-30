import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset
df = pd.read_excel("Dropoutdataset.xlsx")


st.image("Africdsa.jpeg")

# Provide a title for the app
st.title("Multiple Linear Regression App")

st.header('Dataset Concept.', divider='rainbow')

st.write("""The Dropout dataset is a comprehensive collection of information related to students' academic performance and various socio-economic factors, 
            aimed at understanding the factors influencing students decisions to either graduate, dropout, or remain enrolled in educational institutions.
            This dataset includes features such as socio-economic background, parental education, academic scores, attendance,and extracurricular activities.
            In the context of multi-linear regression, researchers and 
            data scientists utilize this dataset to build predictive models that can assess the likelihood of a student either graduating, 
            dropping out, or remaining enrolled based on a combination of these factors. By employing multi-linear regression techniques, 
            the dataset allows for the examination of the relationships and interactions among multiple independent variables simultaneously. 
            The model seeks to identify which specific factors play a significant role in predicting the educational outcomes of students, 
            providing valuable insights for educators, policymakers, and institutions to implement targeted interventions and support systems for at-risk students. 
            Through the analysis of the Dropout dataset, it becomes possible to develop more informed strategies to improve overall student success and reduce dropout rates.""")


# ---------------------------------Display EDA info of the dataset ------------------------------------------------

st.header('Exploratory Data Analysis (EDA).', divider='rainbow')
#st.header('_Streamlit_ is :blue[cool] :sunglasses:')

if st.checkbox("Dataset Info"):
    st.write("Dataset info:", df.info())

if st.checkbox("Number of Rows"):
    st.write("Number of rows:", df.shape[0])

if st.checkbox("Number of Columns"):
    st.write("Number of columns:", df.shape[1])

if st.checkbox("Column names"):
    st.write("Column names:", df.columns.tolist())
    
if st.checkbox("Data types"):    
    st.write("Data types:", df.dtypes)
    
if st.checkbox("Missing Values"):
    st.write("Missing values:", df.isnull().sum())

if st.checkbox("Statistical Summary"):
    st.write("Statistical Summary:", df.describe())
    

#------------------------------------------Visualization------------------------------------------------------------ 

st.header('Visualization of the Dataset (VIZ).', divider='rainbow')


# Create the Bar chart
if st.checkbox("Inflation Rate Bar Chart"):
    st.write("Bar Chart for Inflation Rate Against GDP")
    st.bar_chart(x="Inflation rate", y="GDP", data=df , color=["#FF0000"])

# Create the Bar chart
if st.checkbox("Gender Bar Chart"):
    st.write("Bar Chart for Gender Against GDP")
    st.bar_chart(x="Gender",  y="GDP",data=df , color=["#FF0000"])


# create a Line chart

if st.checkbox("Inflation Rate line Chart"):
    st.write("Line Chart for Inflation Rate Against GDP")
    st.line_chart(x="Inflation rate", y="GDP", data=df , color=["#ffaa0088"])


# create a histogram

if st.checkbox("Scatter Plot"):
    st.write("Scatter Chart of GDP Against Target")
    # Create the histogram using Altair
    st.scatter_chart(
    x="Target",
    y='GDP',
    data = df,
    color=["#ffaa0088"]
    )
    


#------------------------------Multiple Linear Regression model ----------------------------------

# Encode target column using LabelEncoder
university = LabelEncoder()
df['Target'] = university.fit_transform(df['Target'])

# Use OneHotEncoder to encode categorical features
ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), ['Target'])], remainder="passthrough")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y_encoded = ct.fit_transform(df[['Target']])

# Split the data into training and testing sets
x_train, X_test, y_train, Y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

# Fit multiple linear regression to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#st.sidebar.image("Africdsa.jpg")

# User Input for Independent Variables
st.sidebar.header("Enter values to be Predicted", divider='rainbow')

# Create input boxes for each feature
user_input = {}
for feature in df.columns[:-1]:  # Exclude the target column
    user_input[feature] = st.sidebar.text_input(f"Enter {feature}")

# Button to trigger prediction
if st.sidebar.button("Predict"):
    # Create a DataFrame from user input
    user_input_df = pd.DataFrame([user_input], dtype=float)

    # Predict using the trained model
    y_pred = regressor.predict(user_input_df)

    # Inverse transform to get the original target values
    predicted_class = university.inverse_transform(np.argmax(y_pred, axis=1))

    # Display the predicted class
    st.header("Predicted Result Outcome:", divider='rainbow')
    st.write(predicted_class[0])
    
    from sklearn.metrics import r2_score
    st.write("Model Accuracy",r2_score(Y_test, y_pred))

