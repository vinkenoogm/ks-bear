import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///local_dev.sqlite")

ADMIN_USERS = os.getenv("ADMIN_USERS", "").split(",")
BEAR_LABELS = ("Trap 1", "Trap 2")
