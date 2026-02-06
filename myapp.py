
# Framework imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.liner_model import LinerRegression
from sklearn.metrics import mean_squared_error,r2_score
# Streamlit UI
st.title("Good to Go, buddy 🚀")
st.subheader("ML with me")
#either csv file
#web application have example..
# create sidebar
st.sidebar.header("upload CSV data or use sample")
user_example = st.sidebar.checkbox("Use example dataset")
#load dataset.......
if user_example:
  df = sns.load_dataset('tips')

  df = df.dropna()
  st.success("loaded sample dataset: 'tips'")
  

else:
  uploaded_file = st.sidebar.file_uploader("Upload your csv file",type=["csv"])
  if uploaded_file:
    df = pd.read_csv(uploaded_file)
  else:
    st.warning("please upload a csv file or use example")
    st.stop()
  #show data
st.subheader("Dataset preview")
st.write(df.head())

#EDA and model training
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
  st.error("nedd atleast two numeric columns for regression.")
  st.stop()

target = st.selectbox("select target variable",numeric_cols)
features = st.multiselect("select input feature columns",[col for col in numeric_cols if col != target], default = [col for col in numeric_cols if col != target])

# Keep all files; filter by choice.
if len(features) == 0:
  st.write("please select atleast one features")
  st.stop()

df[features + [[target]].dropna()

x = df[features]
y = df[target]
#transform//scaing--->> convert -1 to +1

scaler =StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test,y_train, y_test = train_test_split(X_scaled,y,test_size=0.2, random_state = 42)
#train_test_split(X_scaled,y,test_sizee=0.2,random_state = 42)

model = LinerRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
msc = mean_score(y_test, y_pred)

st.subheader("Model Evaludation")
st.write(f"Mean Sqaure Error: {mse:.2f}")
st.write(f"R^2 Score: {r2:.2f}")






        
