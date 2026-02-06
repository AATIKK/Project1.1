
# Framework imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

