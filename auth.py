import os
import streamlit as st


def _admin_password() -> str:
    if "ADMIN_PASSWORD" in st.secrets:
        return st.secrets["ADMIN_PASSWORD"]
    return os.getenv("ADMIN_PASSWORD", "")


def require_admin():
    if st.session_state.get("is_admin"):
        if st.sidebar.button("Log out admin"):
            st.session_state.is_admin = False
            st.rerun()
        return

    st.info("Admin access is required for this page.")
    pwd = st.text_input("Admin password", type="password")
    if st.button("Log in", type="primary"):
        if pwd and pwd == _admin_password():
            st.session_state.is_admin = True
            st.rerun()
        st.error("Invalid password.")
    st.stop()
