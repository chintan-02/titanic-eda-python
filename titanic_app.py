import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🚢 Titanic Survival Analysis App")

# Load data
df = pd.read_csv("titanic.csv")

# Dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Shape
st.write("Shape of dataset:", df.shape)

# Missing values
st.subheader("❗ Missing Values")
st.write(df.isnull().sum())

# Survival count
st.subheader("🧍 Survival Count")
fig1, ax1 = plt.subplots()
sns.countplot(x='Survived', data=df, ax=ax1)
st.pyplot(fig1)

# Age distribution
st.subheader("🎂 Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df['Age'].dropna(), kde=True, ax=ax2)
st.pyplot(fig2)

# Correlation heatmap
st.subheader("🔥 Correlation Matrix")
fig3, ax3 = plt.subplots(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)