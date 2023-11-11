import streamlit as st
import numpy as np
import pandas as pd

# Create a function to calculate cosine similarity between the two vectors

def cosine_similarity(vector1, vector2):
    # Based on https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

# Wide layout
st.set_page_config(layout="wide")

# Using streamlit to input one csv file

uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is None:
    st.stop()

# Read the csv file into a pandas dataframe (columns are: text and embedding)

df = pd.read_csv(uploaded_file)

# Add a dropdown list to select any "text" column from the dataframe
selection = st.selectbox("Select a base text", df["text"].values)

base_vector = eval(df[df["text"] == selection]["embedding"].values[0])

# Create a new dataframe where the selected line is taken out
df2 = df[df["text"] != selection].copy()

# Add a new column which is the cosine similarity between the selected line and the other lines
df2["similarity"] = df2["embedding"].apply(lambda x: cosine_similarity(base_vector, eval(x)))

# Sort the dataframe by the similarity column, descending
df2 = df2.sort_values(by="similarity", ascending=False)

# Display the dataframe (in full width)
st.dataframe(df2[["text", "similarity"]], use_container_width=True)