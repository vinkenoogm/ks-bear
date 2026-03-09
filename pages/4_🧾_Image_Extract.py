import io
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

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
    kingdom_file_idx: int = -1
    alliance_tags: set[str] = field(default_factory=set)
    used_fallback_match: bool = False
    kingdom_ocr_conf: float = 0.0
    alliance_ocr_conf: float = 0.0


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def number_to_int(value: str) -> int:
    return int(re.sub(r"[^\d]", "", value))


def preprocess_variants(image: Image.Image) -> list[Image.Image]:
    base = ImageOps.exif_transpose(image).convert("RGB")
    gray = ImageOps.grayscale(base)
    contrasted = ImageOps.autocontrast(gray)
    upscaled = contrasted.resize((contrasted.width * 2, contrasted.height * 2), Image.Resampling.LANCZOS)
    thresholded = contrasted.point(lambda px: 255 if px > 165 else 0, mode="1")
    return [base, contrasted, upscaled, thresholded.convert("L")]


def ocr_confidence(image: Image.Image) -> float:
    if not pytesseract:
        return 0.0
    data = pytesseract.image_to_data(image, config="--psm 6", output_type=pytesseract.Output.DICT)
    confs: list[float] = []
    for conf in data.get("conf", []):
        try:
            value = float(conf)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            confs.append(value)
    if not confs:
        return 0.0
    return sum(confs) / len(confs)


def ocr_candidates(image: Image.Image) -> list[tuple[str, float]]:
    if not pytesseract:
        return []
    results: list[tuple[str, float]] = []
    for variant in preprocess_variants(image):
        text = pytesseract.image_to_string(variant, config="--psm 6")
        cleaned = text.strip()
        if cleaned:
            results.append((cleaned, ocr_confidence(variant)))
    return results


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


def extract_tags(value: str) -> set[str]:
    tags = {tag.upper() for tag in re.findall(r"\[([A-Za-z0-9]{2,6})\]", value)}
    tags |= {tag.upper() for tag in re.findall(r"\[([A-Za-z0-9]{2,6})(?=[A-Za-z0-9])", value)}
    return tags


def normalize_alliance_name(name: str) -> str:
    cleaned = normalize_spaces(name)
    cleaned = cleaned.replace("]", "] ")
    cleaned = re.sub(r"[^A-Za-z0-9\[\] ]+", " ", cleaned)
    cleaned = normalize_spaces(cleaned)
    cleaned = re.sub(r"\[([A-Za-z0-9]{2,6})J(?=[A-Za-z])", r"[\1]", cleaned)
    cleaned = re.sub(r"\[([A-Za-z0-9]{2,6})(?!\])", r"[\1]", cleaned)

    pos = cleaned.find("[")
    if pos >= 0:
        cleaned = cleaned[pos:]
    cleaned = normalize_spaces(cleaned)

    m = re.match(r"\[([A-Za-z0-9]{2,6})\]\s*(.*)", cleaned)
    if m:
        tag = m.group(1).upper()
        tail = m.group(2).replace(" ", "")
        return f"[{tag}]{tail.upper()}".strip()
    return cleaned.upper()


def extract_alliance_power(text: str) -> dict[str, Any]:
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
        if score < 100000:
            continue
        name = strip_leading_rank(line[: num_match.start()].strip())
        name = normalize_alliance_name(name)
        if not name:
            continue
        parsed.append((name, score))

    top3 = parsed[:3]
    if len(top3) < 3:
        return {"combined_power_top3": "", "top3_alliances": "", "alliance_tags": set()}

    combined_power = str(sum(score for _, score in top3))
    top3_names = " | ".join(name for name, _ in top3)
    top3_tags = set()
    for name, _ in top3:
        top3_tags |= extract_tags(name)
    return {"combined_power_top3": combined_power, "top3_alliances": top3_names, "alliance_tags": top3_tags}


def split_alliance_and_tag(governor_name: str) -> tuple[str, str]:
    cleaned = normalize_spaces(governor_name)
    cleaned = re.sub(r"\[([A-Za-z0-9]{2,6})J(?=[A-Za-z])", r"[\1]", cleaned)
    alliance_match = re.search(r"\[([A-Za-z0-9]{2,6})\]", cleaned)
    alliance = f"[{alliance_match.group(1).upper()}]" if alliance_match else ""
    gamer_tag = re.sub(r"\[[A-Za-z0-9]{2,6}\]", "", cleaned).strip()
    return alliance, gamer_tag


def extract_mystic_rows(text: str) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    tags: set[str] = set()
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
        tags |= extract_tags(alliance)

        rows.append(
            {
                "Alliance": alliance,
                "Mystic trial": str(score),
                "Gamer tag": gamer_tag,
            }
        )
    return rows, tags


def best_context(contexts: list[Context], tags: set[str]) -> Context | None:
    if not contexts:
        return None

    best = contexts[-1]
    best_score = -1
    for ctx in reversed(contexts):
        score = len(tags & ctx.alliance_tags) if tags else 0
        if score > best_score:
            best = ctx
            best_score = score
    return best


def choose_best_candidate(
    candidates: list[tuple[str, float]], extractor
) -> tuple[str, float, Any]:
    best_text = ""
    best_conf = 0.0
    best_parsed = None
    best_score = -1.0
    for text, conf in candidates:
        parsed = extractor(text)
        if isinstance(parsed, dict):
            score = 0.0
            if parsed.get("kingdom"):
                score += 2
            if parsed.get("transfers_used"):
                score += 1
            if parsed.get("combined_power_top3"):
                score += 2
            if parsed.get("top3_alliances"):
                score += 2
            tags = parsed.get("alliance_tags")
            if isinstance(tags, set):
                score += min(len(tags), 3)
        elif isinstance(parsed, tuple):
            rows, tags = parsed
            score = len(rows) * 2 + min(len(tags), 3)
        else:
            score = 0.0

        score += conf / 100
        if score > best_score:
            best_score = score
            best_text = text
            best_conf = conf
            best_parsed = parsed

    return best_text, best_conf, best_parsed


def run_extraction(files: list[Any]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    contexts: list[Context] = []
    all_rows: list[dict[str, Any]] = []
    file_meta: list[dict[str, str]] = []

    for idx, uploaded_file in enumerate(files):
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        candidates = ocr_candidates(image)
        combined_text = "\n".join(text for text, _ in candidates)
        screen_type = classify_text(combined_text)

        if screen_type == "kingdom":
            best_text, best_conf, parsed = choose_best_candidate(candidates, extract_kingdom)
            data = parsed if isinstance(parsed, dict) else extract_kingdom(best_text)
            ctx = Context(
                kingdom=data["kingdom"],
                transfers_used=data["transfers_used"],
                kingdom_file_idx=idx,
                kingdom_ocr_conf=best_conf,
            )
            contexts.append(ctx)
            file_meta.append(
                {
                    "file": uploaded_file.name,
                    "type": screen_type,
                    "matched_kingdom": ctx.kingdom,
                    "ocr_confidence": f"{best_conf:.1f}",
                }
            )
        elif screen_type == "alliance_power":
            best_text, best_conf, parsed = choose_best_candidate(candidates, extract_alliance_power)
            data = parsed if isinstance(parsed, dict) else extract_alliance_power(best_text)
            tags = data["alliance_tags"]
            ctx = best_context(contexts=contexts, tags=tags)
            if ctx:
                overlap = len(tags & ctx.alliance_tags)
                ctx.combined_power_top3 = data["combined_power_top3"] or ctx.combined_power_top3
                ctx.top3_alliances = data["top3_alliances"] or ctx.top3_alliances
                ctx.alliance_tags |= tags
                ctx.alliance_ocr_conf = max(ctx.alliance_ocr_conf, best_conf)
                if overlap == 0 and tags:
                    ctx.used_fallback_match = True
            file_meta.append(
                {
                    "file": uploaded_file.name,
                    "type": screen_type,
                    "matched_kingdom": ctx.kingdom if ctx else "",
                    "ocr_confidence": f"{best_conf:.1f}",
                }
            )
        elif screen_type == "mystic_trial":
            best_text, best_conf, parsed = choose_best_candidate(candidates, extract_mystic_rows)
            if isinstance(parsed, tuple):
                parsed_rows, tags = parsed
            else:
                parsed_rows, tags = extract_mystic_rows(text=best_text)
            ctx = best_context(contexts=contexts, tags=tags)
            if ctx:
                overlap = len(tags & ctx.alliance_tags)
                ctx.alliance_tags |= tags
                if overlap == 0 and tags:
                    ctx.used_fallback_match = True
            for parsed in parsed_rows:
                manual_check = False
                reasons: list[str] = []
                if best_conf < 60:
                    manual_check = True
                    reasons.append("low mystic OCR confidence")
                if not ctx or not ctx.kingdom:
                    manual_check = True
                    reasons.append("no kingdom match")
                if ctx and ctx.used_fallback_match:
                    manual_check = True
                    reasons.append("fallback kingdom match")
                if not parsed["Alliance"] or not parsed["Gamer tag"]:
                    manual_check = True
                    reasons.append("missing alliance or gamer tag")
                if ctx and (not ctx.top3_alliances or not ctx.combined_power_top3):
                    manual_check = True
                    reasons.append("missing alliance power context")
                if ctx and (ctx.kingdom_ocr_conf < 60 or ctx.alliance_ocr_conf < 55):
                    manual_check = True
                    reasons.append("low context OCR confidence")

                all_rows.append(
                    {
                        "Kingdom": ctx.kingdom if ctx else "",
                        "Transfers used": ctx.transfers_used if ctx else "",
                        "Combined power of top 3 alliances": ctx.combined_power_top3 if ctx else "",
                        "Top 3 alliances": ctx.top3_alliances if ctx else "",
                        "Alliance": parsed["Alliance"],
                        "Mystic trial": parsed["Mystic trial"],
                        "Gamer tag": parsed["Gamer tag"],
                        "Needs manual check": "yes" if manual_check else "no",
                        "Manual check reason": "; ".join(sorted(set(reasons))),
                        "Source file": uploaded_file.name,
                    }
                )
            file_meta.append(
                {
                    "file": uploaded_file.name,
                    "type": screen_type,
                    "matched_kingdom": ctx.kingdom if ctx else "",
                    "ocr_confidence": f"{best_conf:.1f}",
                }
            )
        else:
            best_conf = max((conf for _, conf in candidates), default=0.0)
            file_meta.append(
                {
                    "file": uploaded_file.name,
                    "type": screen_type,
                    "matched_kingdom": "",
                    "ocr_confidence": f"{best_conf:.1f}",
                }
            )

    columns = [
        "Kingdom",
        "Transfers used",
        "Combined power of top 3 alliances",
        "Top 3 alliances",
        "Alliance",
        "Mystic trial",
        "Gamer tag",
        "Needs manual check",
        "Manual check reason",
        "Source file",
    ]
    if not all_rows:
        return pd.DataFrame(columns=columns), file_meta
    return pd.DataFrame(all_rows, columns=columns), file_meta


st.title("Image Extract")

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
