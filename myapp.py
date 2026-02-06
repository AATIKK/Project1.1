import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# FIX 1: Spelled "Linear", not "Liner"
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit UI
st.title("Good to Go, buddy 🚀")
st.subheader("ML with me")

# create sidebar
st.sidebar.header("upload CSV data or use sample")
user_example = st.sidebar.checkbox("Use example dataset")

# load dataset
if user_example:
    df = sns.load_dataset('tips')
    df = df.dropna()
    st.success("loaded sample dataset: 'tips'")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your csv file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("please upload a csv file or use example")
        st.stop()

# show data
st.subheader("Dataset preview")
st.write(df.head())

# EDA and model training
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("need atleast two numeric columns for regression.")
    st.stop()

target = st.selectbox("select target variable", numeric_cols)
features = st.multiselect("select input feature columns", [col for col in numeric_cols if col != target], default=[col for col in numeric_cols if col != target])

if len(features) == 0:
    st.write("please select atleast one features")
    st.stop()

# Cleaning data
df = df[features + [target]].dropna()

# FIX 2: Ensure variable names are consistent (X vs x)
X = df[features] 
y = df[target]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# FIX 3: Spelled "LinearRegression"
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# FIX 4: Correct function name and variables
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"Mean Square Error: {mse:.2f}")
st.write(f"R^2 Score: {r2:.2f}")
st.subheader("Make a Prediction")
input_data = {}
valid_input = True
for feature in features:
    user_val = st.text_input(f"Enter {feature} (numeric value)")
    try:
        if user_val.strip() == "":
            valid_input = False
        else:
            input_data[feature] = float(user_val)

    except valueError:
        valid_input = False

