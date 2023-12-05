import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.title('Regression Model Web App')

# Load the dataset
df = pd.read_csv(r'C:\Users\HP Pavilion\Desktop\Demo\50_Startups.csv')

# Sidebar for user input
st.sidebar.header('User Input Features')

# Display a random row from the dataset
st.sidebar.subheader('Sample Data')
st.sidebar.write(df.sample(1))

# Univariate Analysis
st.sidebar.header('Univariate Analysis')

# Histograms
st.sidebar.subheader('Histograms')
selected_column = st.sidebar.selectbox('Select a column for histogram:', df.columns)
plt.figure(figsize=(8, 6))
sns.histplot(df[selected_column], kde=True, bins=20, color='skyblue')
plt.title(f'Histogram of {selected_column}')
plt.xlabel(selected_column)
plt.ylabel('Frequency')
st.pyplot()

# Box Plots
st.sidebar.subheader('Box Plots')
selected_column_box = st.sidebar.selectbox('Select a column for box plot:', df.columns)
plt.figure(figsize=(8, 6))
sns.boxplot(x=df[selected_column_box], color='lightcoral')
plt.title(f'Box Plot of {selected_column_box}')
plt.xlabel(selected_column_box)
st.pyplot()

# Violin Plots
st.sidebar.subheader('Violin Plots')
selected_column_violin = st.sidebar.selectbox('Select a column for violin plot:', df.columns)
plt.figure(figsize=(8, 6))
sns.violinplot(x=df[selected_column_violin], color='mediumseagreen')
plt.title(f'Violin Plot of {selected_column_violin}')
plt.xlabel(selected_column_violin)
st.pyplot()

# Main content
st.header('Regression Model')

# Preprocessing
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

states = pd.get_dummies(x['State'], drop_first=True)
states = states.astype(int)

x = x.drop('State', axis=1)
x = pd.concat([x, states], axis=1)

# Model fitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lin_regressor = LinearRegression()
lin_regressor.fit(x_train, y_train)

# Prediction
st.subheader('Make Predictions')
sample_data = df.sample(1).iloc[:, :-1]
sample_data_states = pd.get_dummies(sample_data['State'], drop_first=True)
sample_data_states = sample_data_states.astype(int)
sample_data = sample_data.drop('State', axis=1)
sample_data = pd.concat([sample_data, sample_data_states], axis=1)

prediction_button = st.button('Predict')
if prediction_button:
    prediction = lin_regressor.predict(sample_data)
    st.write(f'Predicted Profit: {prediction[0]}')

# Model evaluation
st.header('Model Evaluation')

# Display R2 score and Mean Squared Error
y_pred = lin_regressor.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.write(f'R2 Score: {r2}')
st.write(f'Mean Squared Error: {mse}')
