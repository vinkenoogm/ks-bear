import io
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from auth import require_admin
from config.settings import TESSERACT_CMD

try:
    import pytesseract
except ModuleNotFoundError:
    pytesseract = None

if pytesseract and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


@dataclass
class Context:
    kingdom: str = ""
    transfers_used: str = ""
    combined_power_top3: str = ""
    top3_alliances: str = ""


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def number_to_int(value: str) -> int:
    return int(re.sub(r"[^\d]", "", value))


def preprocess_variants(image: Image.Image) -> list[Image.Image]:
    gray = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(gray)
    thresholded = contrasted.point(lambda px: 255 if px > 165 else 0, mode="1")
    return [image, contrasted, thresholded.convert("L")]


def ocr_text(image: Image.Image) -> str:
    if not pytesseract:
        return ""
    chunks: list[str] = []
    for variant in preprocess_variants(image):
        text = pytesseract.image_to_string(variant, config="--psm 6")
        cleaned = text.strip()
        if cleaned:
            chunks.append(cleaned)
    return "\n".join(chunks)


def classify_text(text: str) -> str:
    lower = text.lower()
    if "kingdom details" in lower:
        return "kingdom"
    if "alliance power" in lower:
        return "alliance_power"
    if "mystic trial" in lower:
        return "mystic_trial"
    return "unknown"


def extract_kingdom(text: str) -> dict[str, str]:
    kingdom_match = re.search(r"Kingdom\s*#?\s*(\d+)", text, flags=re.IGNORECASE)
    transfer_match = re.search(
        r"Transfer\s*Req\.?\s*\((\d+)\s*/\s*\d+\)", text, flags=re.IGNORECASE
    )
    return {
        "kingdom": kingdom_match.group(1) if kingdom_match else "",
        "transfers_used": transfer_match.group(1) if transfer_match else "",
    }


def trailing_number(line: str) -> re.Match[str] | None:
    return re.search(r"([0-9][0-9,\.]{2,})\s*$", line)


def strip_leading_rank(text: str) -> str:
    return re.sub(r"^\s*[0-9Il]{1,2}\s*", "", text).strip()


def extract_alliance_power(text: str) -> dict[str, str]:
    parsed: list[tuple[str, int]] = []
    for raw in text.splitlines():
        line = normalize_spaces(raw)
        if not line:
            continue
        if any(word in line.lower() for word in ("ranking", "alliance", "power")):
            continue

        num_match = trailing_number(line)
        if not num_match:
            continue
        score = number_to_int(num_match.group(1))
        name = strip_leading_rank(line[: num_match.start()].strip())
        if not name:
            continue
        parsed.append((name, score))

    top3 = parsed[:3]
    if len(top3) < 3:
        return {"combined_power_top3": "", "top3_alliances": ""}

    combined_power = str(sum(score for _, score in top3))
    top3_names = " | ".join(name for name, _ in top3)
    return {"combined_power_top3": combined_power, "top3_alliances": top3_names}


def split_alliance_and_tag(governor_name: str) -> tuple[str, str]:
    cleaned = normalize_spaces(governor_name)
    alliance_match = re.search(r"(\[[A-Za-z0-9]{2,6}\])", cleaned)
    alliance = alliance_match.group(1) if alliance_match else ""
    gamer_tag = re.sub(r"\[[A-Za-z0-9]{2,6}\]", "", cleaned).strip()
    return alliance, gamer_tag


def extract_mystic_rows(text: str, context: Context, file_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = normalize_spaces(raw)
        if not line:
            continue
        lower = line.lower()
        if any(word in lower for word in ("ranking", "governor", "total stages", "mystic trial")):
            continue

        num_match = trailing_number(line)
        if not num_match:
            continue
        score = number_to_int(num_match.group(1))
        if score <= 1800:
            continue

        governor = strip_leading_rank(line[: num_match.start()].strip())
        alliance, gamer_tag = split_alliance_and_tag(governor)

        rows.append(
            {
                "Kingdom": context.kingdom,
                "Transfers used": context.transfers_used,
                "Combined power of top 3 alliances": context.combined_power_top3,
                "Top 3 alliances": context.top3_alliances,
                "Alliance": alliance,
                "Mystic trial": str(score),
                "Gamer tag": gamer_tag,
                "Source file": file_name,
            }
        )
    return rows


def run_extraction(files: list[Any]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    context = Context()
    all_rows: list[dict[str, Any]] = []
    file_meta: list[dict[str, str]] = []

    for uploaded_file in sorted(files, key=lambda f: f.name.lower()):
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        text = ocr_text(image)
        screen_type = classify_text(text)
        file_meta.append({"file": uploaded_file.name, "type": screen_type})

        if screen_type == "kingdom":
            data = extract_kingdom(text)
            context.kingdom = data["kingdom"] or context.kingdom
            context.transfers_used = data["transfers_used"] or context.transfers_used
        elif screen_type == "alliance_power":
            data = extract_alliance_power(text)
            context.combined_power_top3 = data["combined_power_top3"] or context.combined_power_top3
            context.top3_alliances = data["top3_alliances"] or context.top3_alliances
        elif screen_type == "mystic_trial":
            all_rows.extend(extract_mystic_rows(text=text, context=context, file_name=uploaded_file.name))

    columns = [
        "Kingdom",
        "Transfers used",
        "Combined power of top 3 alliances",
        "Top 3 alliances",
        "Alliance",
        "Mystic trial",
        "Gamer tag",
        "Source file",
    ]
    if not all_rows:
        return pd.DataFrame(columns=columns), file_meta
    return pd.DataFrame(all_rows, columns=columns), file_meta


st.title("Image Extract")
require_admin()

st.caption(
    "Upload any number of screenshots. Rows are created for Mystic Trial entries above 1800."
)

uploaded_files = st.file_uploader(
    "Screenshots",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if st.button("Extract data", type="primary"):
    if not uploaded_files:
        st.warning("Upload at least one screenshot.")
        st.stop()

    if not pytesseract:
        st.error("`pytesseract` is not installed. Run `pip install -r requirements.txt` first.")
        st.stop()

    try:
        df, meta = run_extraction(uploaded_files)
    except Exception as exc:
        if pytesseract and isinstance(exc, pytesseract.TesseractNotFoundError):
            st.error(
                "Tesseract OCR was not found. Install Tesseract and set TESSERACT_CMD in environment if needed."
            )
            st.stop()
        raise

    st.subheader("Detected files")
    st.dataframe(pd.DataFrame(meta), use_container_width=True, hide_index=True)

    st.subheader("Extracted rows")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download extracted.csv",
        data=csv_data,
        file_name="extracted.csv",
        mime="text/csv",
    )
