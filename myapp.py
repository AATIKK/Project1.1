
# Framework imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Streamlit UI
st.title("Good to Go, buddy 🚀")
st.subheader("ML with me")
#either csv file
#web application have example..
# create sidebar
st.sidebar.header("upload CSV data or use sample")
