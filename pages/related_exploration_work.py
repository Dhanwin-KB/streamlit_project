import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('topten.csv', encoding='latin-1')
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.markdown('### Summary')
average_duration = df['dur'].mean()
most_popular_song = df.loc[df['pop'].idxmax()]
top_genre = df['top genre'].value_counts().idxmax()

col1, col2, col3 = st.columns(3)
col1.metric("Average Duration", f"{average_duration:.2f} seconds")
col2.metric("Most Popular Song", f"{most_popular_song['title']}")
col3.metric("Top Genre", top_genre)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Length vs. Popularity")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='dur', y='pop', alpha=0.5)
    ax1.set_xlabel("Length (seconds)")
    ax1.set_ylabel("Popularity")
    ax1.set_title("Length vs. Popularity")
    st.pyplot(fig1)

with c2:    
    st.subheader("Year vs. Popularity")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, x='year', y='pop', ci=None)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Average Popularity")
    ax2.set_title("Average Popularity by Year")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

st.subheader("Top Ten Songs")
st.dataframe(df)

st.subheader("3D Plot: Energy, Danceability, and Valence")
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(df['nrgy'], df['dnce'], df['val'], alpha=0.5)
ax3.set_xlabel("Energy")
ax3.set_ylabel("Danceability")
ax3.set_zlabel("Valence")
ax3.set_title("3D Plot: Energy, Danceability, and Valence")
st.pyplot(fig3)
