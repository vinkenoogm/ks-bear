import os
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
except Exception:
    st = None


def get_setting(name: str, default=None):
    if st is not None:
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            pass
    return os.getenv(name, default)


DATABASE_URL = get_setting("DATABASE_URL", "sqlite:///local_dev.sqlite")

ADMIN_USERS = str(get_setting("ADMIN_USERS", "")).split(",")
BEAR_LABELS = ("Bear Trap 1", "Bear Trap 2")
TESSERACT_CMD = get_setting("TESSERACT_CMD", "")

STRATO_SFTP_HOST = get_setting("STRATO_SFTP_HOST", "")
STRATO_SFTP_PORT = int(get_setting("STRATO_SFTP_PORT", 22))
STRATO_SFTP_USER = get_setting("STRATO_SFTP_USER", "")
STRATO_SFTP_PASSWORD = get_setting("STRATO_SFTP_PASSWORD", "")
STRATO_UPLOAD_BASE_PATH = get_setting("STRATO_UPLOAD_BASE_PATH", "/bear-images")
STRATO_PUBLIC_BASE_URL = get_setting("STRATO_PUBLIC_BASE_URL", "")

try:
    OCR_TIMEOUT_SECONDS = float(get_setting("OCR_TIMEOUT_SECONDS", "4"))
except ValueError:
    OCR_TIMEOUT_SECONDS = 4.0
