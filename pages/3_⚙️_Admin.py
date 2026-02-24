import streamlit as st
from auth import require_admin

st.title("⚙️ Admin")

require_admin()

st.write("Admin-only configuration panel.")