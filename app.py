import streamlit as st
from db import init_db

st.set_page_config(page_title="Kingshot Bear Tracker", layout="wide")

init_db()

st.title("🐻 Kingshot Bear Tracker")

st.markdown("""
Welcome to the official infrastructure for Extremely Serious Bear Events.
Use the sidebar to navigate.
""")