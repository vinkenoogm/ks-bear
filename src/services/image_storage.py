import io
import os
import posixpath
from datetime import date
from uuid import uuid4

from config.settings import (
    STRATO_PUBLIC_BASE_URL,
    STRATO_SFTP_HOST,
    STRATO_SFTP_PASSWORD,
    STRATO_SFTP_PORT,
    STRATO_SFTP_USER,
    STRATO_UPLOAD_BASE_PATH,
)


def strato_storage_enabled() -> bool:
    return all([STRATO_SFTP_HOST, STRATO_SFTP_USER, STRATO_SFTP_PASSWORD])


def upload_event_images(event_date: date, bear_label: str, uploaded_files) -> list[dict]:
    if not strato_storage_enabled():
        return []

    import paramiko

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        STRATO_SFTP_HOST,
        port=STRATO_SFTP_PORT,
        username=STRATO_SFTP_USER,
        password=STRATO_SFTP_PASSWORD,
        timeout=20,
    )

    uploaded_rows = []
    try:
        with ssh.open_sftp() as sftp:
            remote_dir = _remote_dir_for_event(event_date, bear_label)
            _ensure_remote_dir(sftp, remote_dir)

            for uploaded_file in uploaded_files:
                uploaded_file.seek(0)
                original_name = os.path.basename(uploaded_file.name)
                remote_name = f"{uuid4().hex}_{original_name}"
                remote_path = posixpath.join(remote_dir, remote_name)
                sftp.putfo(io.BytesIO(uploaded_file.getvalue()), remote_path)
                uploaded_rows.append(
                    {
                        "original_filename": original_name,
                        "storage_path": remote_path,
                        "public_url": _public_url(remote_path),
                    }
                )
    finally:
        ssh.close()

    return uploaded_rows


def _remote_dir_for_event(event_date: date, bear_label: str) -> str:
    safe_label = bear_label.lower().replace(" ", "-")
    base_path = STRATO_UPLOAD_BASE_PATH.strip() or "/bear-images"
    if not base_path.startswith("/"):
        base_path = f"/{base_path}"
    return posixpath.join(base_path, str(event_date), safe_label)


def _ensure_remote_dir(sftp, remote_dir: str):
    current = ""
    for part in remote_dir.strip("/").split("/"):
        current = f"{current}/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def _public_url(remote_path: str):
    if not STRATO_PUBLIC_BASE_URL:
        return None

    base_url = STRATO_PUBLIC_BASE_URL.rstrip("/")
    base_path = STRATO_UPLOAD_BASE_PATH.strip() or "/bear-images"
    if not base_path.startswith("/"):
        base_path = f"/{base_path}"

    if remote_path.startswith(base_path):
        relative_path = remote_path[len(base_path):].lstrip("/")
    else:
        relative_path = remote_path.lstrip("/")

    return f"{base_url}/{relative_path}"
