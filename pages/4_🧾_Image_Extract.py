import io
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from config.settings import OCR_TIMEOUT_SECONDS, TESSERACT_CMD

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
    alliance_comment: str = ""


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


def ocr_confidence(image: Image.Image, psm: int) -> float:
    if not pytesseract:
        return 0.0
    try:
        data = pytesseract.image_to_data(
            image,
            config=f"--oem 3 --psm {psm}",
            output_type=pytesseract.Output.DICT,
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except RuntimeError:
        return 0.0
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


def ocr_text_conf(image: Image.Image, config: str) -> tuple[str, float]:
    if not pytesseract:
        return "", 0.0
    try:
        text = pytesseract.image_to_string(
            image, config=config, timeout=OCR_TIMEOUT_SECONDS
        ).strip()
        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT,
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except RuntimeError:
        return "", 0.0
    confs: list[float] = []
    for conf in data.get("conf", []):
        try:
            value = float(conf)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            confs.append(value)
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    return text, avg_conf


def crop_rel(image: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    w, h = image.size
    box = (int(w * left), int(h * top), int(w * right), int(h * bottom))
    return image.crop(box)


def best_crop_text(crop: Image.Image, configs: list[str]) -> tuple[str, float]:
    best_text = ""
    best_conf = 0.0
    best_score = -1.0
    for variant in preprocess_variants(crop):
        for config in configs:
            text, conf = ocr_text_conf(variant, config)
            cleaned = normalize_spaces(text)
            if not cleaned:
                continue
            score = conf + min(len(cleaned), 30) / 3
            if score > best_score:
                best_score = score
                best_text = cleaned
                best_conf = conf
    return best_text, best_conf


def ocr_candidates(image: Image.Image) -> list[tuple[str, float]]:
    if not pytesseract:
        return []
    results: list[tuple[str, float]] = []
    for variant in preprocess_variants(image):
        for psm in (6, 4, 11):
            try:
                text = pytesseract.image_to_string(
                    variant,
                    config=f"--oem 3 --psm {psm}",
                    timeout=OCR_TIMEOUT_SECONDS,
                )
            except RuntimeError:
                continue
            cleaned = text.strip()
            if cleaned:
                results.append((cleaned, ocr_confidence(variant, psm)))
    return results


def classify_text(text: str) -> str:
    lower = text.lower()
    if re.search(r"(kingdom|kinadom)", lower) and ("details" in lower or "transfer" in lower):
        return "kingdom"
    if "mystic" in lower and "trial" in lower:
        return "mystic_trial"
    if ("alliance power" in lower) or ("alliance" in lower and "ranking" in lower and "power" in lower):
        return "alliance_power"
    return "unknown"


def extract_kingdom(text: str) -> dict[str, str]:
    kingdom_match = re.search(r"Kingdom\s*#?\s*(\d+)", text, flags=re.IGNORECASE)
    transfer_match = re.search(
        r"Transfer\s*Req\.?\s*\((\d+)\s*/\s*\d+\)", text, flags=re.IGNORECASE
    )
    transfers_used = transfer_match.group(1) if transfer_match else ""
    if not transfers_used:
        generic_req = re.search(r"Req\.?\s*\((\d+)\s*/\s*7\)", text, flags=re.IGNORECASE)
        if generic_req:
            transfers_used = generic_req.group(1)
    return {
        "kingdom": kingdom_match.group(1) if kingdom_match else "",
        "transfers_used": transfers_used,
    }


def trailing_number(line: str) -> re.Match[str] | None:
    return re.search(r"([0-9][0-9,\.]{2,})\s*$", line)


def strip_leading_rank(text: str) -> str:
    return re.sub(r"^\s*[0-9Il]{1,2}\s*", "", text).strip()


def extract_tags(value: str) -> set[str]:
    tags = {tag[:3].upper() for tag in re.findall(r"\[([A-Za-z0-9]{2,6})\]", value)}
    tags |= {tag[:3].upper() for tag in re.findall(r"\[([A-Za-z0-9]{2,6})(?=[A-Za-z0-9])", value)}
    return tags


def normalize_alliance_name(name: str) -> str:
    cleaned = normalize_spaces(name)
    cleaned = re.sub(r"^[^[]*", "", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9\[\]\- ]+", " ", cleaned)
    cleaned = normalize_spaces(cleaned)
    m = re.search(r"\[([A-Za-z0-9]{2,6})[\]\|IJl]?\s*([A-Za-z][A-Za-z0-9\-]{1,})", cleaned)
    if m:
        tag = m.group(1)[:3].upper()
        tail = m.group(2).replace(" ", "")
        return f"[{tag}]{tail.upper()}".strip()
    fallback = re.sub(r"\s+", "", cleaned).upper()
    return fallback


def extract_alliance_power_structured(image: Image.Image) -> tuple[dict[str, Any], float]:
    base = ImageOps.exif_transpose(image).convert("RGB")
    row_top, row_bottom, row_count = 0.16, 0.93, 9
    row_h = (row_bottom - row_top) / row_count

    parsed: list[tuple[str, int]] = []
    confs: list[float] = []
    for i in range(row_count):
        y1 = row_top + i * row_h
        y2 = y1 + row_h
        name_crop = crop_rel(base, 0.23, y1, 0.74, y2)
        score_crop = crop_rel(base, 0.70, y1, 0.98, y2)

        raw_name, name_conf = best_crop_text(
            name_crop, ["--oem 3 --psm 7", "--oem 3 --psm 6", "--oem 3 --psm 11"]
        )
        raw_score, score_conf = best_crop_text(
            score_crop,
            [
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,.",
                "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,.",
            ],
        )
        num_match = re.search(r"\d[\d,\.]{3,}", raw_score)
        if not num_match:
            continue
        score = number_to_int(num_match.group(0))
        if score < 100000:
            continue
        name = normalize_alliance_name(raw_name)
        if not name:
            continue
        parsed.append((name, score))
        confs.extend([name_conf, score_conf])

    top3 = parsed[:3]
    if len(top3) < 3:
        return {"combined_power_top3": "", "top3_alliances": "", "alliance_tags": set(), "ocr_comment": ""}, 0.0

    top3_names = [name for name, _ in top3]
    top3_scores = [score for _, score in top3]
    top3_tags = set()
    for name in top3_names:
        top3_tags |= extract_tags(name)

    ocr_comment = ""
    if len(top3_tags) < 3:
        ocr_comment = "top 3 alliance tags uncertain"

    return {
        "combined_power_top3": str(sum(top3_scores)),
        "top3_alliances": " | ".join(top3_names),
        "alliance_tags": top3_tags,
        "ocr_comment": ocr_comment,
    }, (sum(confs) / len(confs) if confs else 0.0)


def extract_alliance_power(text: str) -> dict[str, Any]:
    parsed: list[tuple[str, int]] = []
    ordered_names: list[str] = []
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
        if name not in ordered_names:
            ordered_names.append(name)

    dedup: dict[int, str] = {}
    for name, score in parsed:
        if score not in dedup or len(name) > len(dedup[score]):
            dedup[score] = name
    scored = [(name, score) for score, name in sorted(dedup.items(), key=lambda x: x[0], reverse=True)]
    if len(scored) < 3 and len(ordered_names) < 3:
        return {"combined_power_top3": "", "top3_alliances": "", "alliance_tags": set(), "ocr_comment": ""}

    top3_names = ordered_names[:3] if len(ordered_names) >= 3 else [name for name, _ in scored[:3]]
    score_by_tag: dict[str, int] = {}
    for name, score in scored:
        tags = extract_tags(name)
        if tags:
            score_by_tag[next(iter(tags))] = score

    top3_scores: list[int] = []
    missing_power = False
    for name in top3_names:
        tags = extract_tags(name)
        if not tags:
            missing_power = True
            continue
        tag = next(iter(tags))
        if tag in score_by_tag:
            top3_scores.append(score_by_tag[tag])
        else:
            missing_power = True

    combined_power = str(sum(top3_scores)) if top3_scores and not missing_power else ""
    top3_names_str = " | ".join(top3_names)
    ocr_comment = "top 3 alliance power incomplete" if missing_power else ""
    top3_tags = set()
    for name in top3_names:
        top3_tags |= extract_tags(name)
    return {
        "combined_power_top3": combined_power,
        "top3_alliances": top3_names_str,
        "alliance_tags": top3_tags,
        "ocr_comment": ocr_comment,
    }


def clean_gamer_tag(raw: str) -> tuple[str, str]:
    cleaned = normalize_spaces(raw)
    tokens = re.findall(r"[A-Za-z0-9]{2,}", cleaned)
    if tokens:
        # Preserve likely multi-word names like "Saint Kaiser".
        if len(tokens) >= 2 and all(re.fullmatch(r"[A-Za-z]{2,}", t) for t in tokens[:2]):
            cleaned = f"{tokens[0]}{tokens[1]}"
        else:
            cleaned = max(tokens, key=len)
    else:
        cleaned = re.sub(r"[^A-Za-z0-9_\-\.]+", "", cleaned)
    if len(cleaned) < 3:
        return "*", "gamer tag uncertain"
    numeric_fix = re.fullmatch(r"[Ss]([0-9]{2,})", cleaned)
    if numeric_fix:
        cleaned = numeric_fix.group(1)
    return cleaned, ""


def split_alliance_and_tag(governor_name: str) -> tuple[str, str, str]:
    cleaned = normalize_spaces(governor_name)
    cleaned = re.sub(r"\[([A-Za-z0-9]{2,6})J(?=[A-Za-z])", r"[\1]", cleaned)
    cleaned = re.sub(r"\[([A-Za-z0-9]{2,6})[Il|](?=[A-Za-z0-9])", r"[\1]", cleaned)
    alliance_match = re.search(r"\[([A-Za-z0-9]{2,6})\]", cleaned)
    alliance = f"[{alliance_match.group(1)[:3].upper()}]" if alliance_match else ""
    gamer_raw = re.sub(r"\[[A-Za-z0-9]{2,6}\]", "", cleaned).strip()
    gamer_tag, comment = clean_gamer_tag(gamer_raw)
    return alliance, gamer_tag, comment


def extract_mystic_rows(text: str) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    tags: set[str] = set()
    lines = [normalize_spaces(raw) for raw in text.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        lower = line.lower()
        if any(word in lower for word in ("ranking", "governor", "total stages", "mystic trial")):
            i += 1
            continue

        num_match = trailing_number(line)
        governor = ""
        if num_match:
            score = number_to_int(num_match.group(1))
            left = strip_leading_rank(line[: num_match.start()].strip())
            if left and re.search(r"[A-Za-z]", left):
                governor = left
            else:
                # If score is alone on its line, backtrack to nearest likely governor line.
                for j in range(i - 1, max(-1, i - 4), -1):
                    prev = lines[j]
                    if not prev:
                        continue
                    prev_lower = prev.lower()
                    if any(word in prev_lower for word in ("ranking", "governor", "total stages", "mystic trial")):
                        continue
                    if trailing_number(prev):
                        continue
                    if re.search(r"[A-Za-z]", prev):
                        governor = strip_leading_rank(prev)
                        break
            i += 1
        else:
            # OCR sometimes puts governor on one line and score on the next.
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            next_num_match = trailing_number(next_line) if next_line else None
            if next_num_match and re.search(r"[A-Za-z]", line):
                governor = strip_leading_rank(line)
                score = number_to_int(next_num_match.group(1))
                i += 2
            else:
                i += 1
                continue

        if score <= 1800:
            continue

        if not governor:
            continue

        alliance, gamer_tag, comment = split_alliance_and_tag(governor)
        tags |= extract_tags(alliance)

        rows.append(
            {
                "Alliance": alliance,
                "Mystic trial": str(score),
                "Gamer tag": gamer_tag,
                "OCR comment": comment,
            }
        )
    return rows, tags


def extract_mystic_rows_structured(image: Image.Image) -> tuple[list[dict[str, str]], set[str], float]:
    base = ImageOps.exif_transpose(image).convert("RGB")
    row_top, row_bottom, row_count = 0.16, 0.93, 9
    row_h = (row_bottom - row_top) / row_count

    rows: list[dict[str, str]] = []
    tags: set[str] = set()
    confs: list[float] = []

    for i in range(row_count):
        y1 = row_top + i * row_h
        y2 = y1 + row_h
        governor_crop = crop_rel(base, 0.23, y1, 0.74, y2)
        score_crop = crop_rel(base, 0.70, y1, 0.98, y2)

        governor_text, governor_conf = best_crop_text(
            governor_crop, ["--oem 3 --psm 7", "--oem 3 --psm 6", "--oem 3 --psm 11"]
        )
        score_text, score_conf = best_crop_text(
            score_crop,
            [
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,.",
                "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,.",
            ],
        )
        confs.extend([governor_conf, score_conf])

        num_match = re.search(r"\d[\d,\.]{2,}", score_text)
        if not num_match:
            continue
        score = number_to_int(num_match.group(0))
        if score <= 1800:
            continue

        alliance, gamer_tag, comment = split_alliance_and_tag(governor_text)
        tags |= extract_tags(alliance)
        rows.append(
            {
                "Alliance": alliance,
                "Mystic trial": str(score),
                "Gamer tag": gamer_tag,
                "OCR comment": comment,
                "_source": "structured",
            }
        )

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return rows, tags, avg_conf


def extract_best_alliance_power(candidates: list[tuple[str, float]]) -> tuple[dict[str, Any], float]:
    best = {"combined_power_top3": "", "top3_alliances": "", "alliance_tags": set(), "ocr_comment": ""}
    best_conf = 0.0
    best_score = -1.0
    for text, conf in candidates:
        parsed = extract_alliance_power(text)
        tags = parsed["alliance_tags"]
        if not parsed["combined_power_top3"]:
            continue
        score = len(tags) * 2 + conf / 20
        score += number_to_int(parsed["combined_power_top3"]) / 10_000_000_000
        if score > best_score:
            best = parsed
            best_conf = conf
            best_score = score
    return best, best_conf


def extract_best_mystic_rows(candidates: list[tuple[str, float]]) -> tuple[list[dict[str, str]], set[str], float]:
    by_score: dict[str, tuple[dict[str, str], float]] = {}
    tags: set[str] = set()
    for text, conf in candidates:
        rows, row_tags = extract_mystic_rows(text)
        tags |= row_tags
        for row in rows:
            score = row["Mystic trial"]
            quality = 0.0
            if row["Alliance"]:
                quality += 2
            if row["Gamer tag"] != "*":
                quality += 2
            if not row["OCR comment"]:
                quality += 1
            quality += conf / 25
            existing = by_score.get(score)
            if not existing or quality > existing[1]:
                chosen = dict(row)
                chosen["_source"] = "fallback"
                by_score[score] = (chosen, quality)

    selected = [data[0] for _, data in sorted(by_score.items(), key=lambda x: int(x[0]), reverse=True)]
    if not selected:
        return [], tags, 0.0

    avg_conf = sum(conf for _, conf in candidates) / len(candidates)
    return selected, tags, avg_conf


def merge_mystic_rows(
    structured_rows: list[dict[str, str]],
    fallback_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_score: dict[str, tuple[dict[str, str], float]] = {}

    def row_quality(row: dict[str, str]) -> float:
        quality = 0.0
        if row.get("Alliance"):
            quality += 2
        if row.get("Gamer tag") and row.get("Gamer tag") != "*":
            quality += 2
        if not row.get("OCR comment"):
            quality += 1
        if row.get("_source") == "structured":
            quality += 0.5
        return quality

    def should_replace(existing: dict[str, str], candidate: dict[str, str], existing_q: float, cand_q: float) -> bool:
        if cand_q > existing_q:
            return True
        if cand_q < existing_q:
            return False

        e_name = (existing.get("Gamer tag") or "").lower()
        c_name = (candidate.get("Gamer tag") or "").lower()
        if e_name == "*" and c_name != "*":
            return True
        if c_name == "*" and e_name != "*":
            return False
        if e_name and c_name and (e_name.startswith(c_name) or c_name.startswith(e_name)):
            return len(c_name) < len(e_name)
        return False

    for row in structured_rows + fallback_rows:
        score = row.get("Mystic trial", "")
        if not score:
            continue
        quality = row_quality(row)
        existing = by_score.get(score)
        if not existing or should_replace(existing[0], row, existing[1], quality):
            by_score[score] = (row, quality)

    merged = [data[0] for _, data in sorted(by_score.items(), key=lambda x: int(x[0]), reverse=True)]
    for row in merged:
        row.pop("_source", None)
    return merged


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


def alliance_data_complete(data: dict[str, Any]) -> bool:
    power = number_to_int(data.get("combined_power_top3", "0") or "0")
    return bool(data.get("combined_power_top3")) and power >= 5_000_000_000 and len(
        data.get("alliance_tags", set())
    ) >= 2


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
            if not data.get("transfers_used"):
                for text, _ in candidates:
                    candidate_data = extract_kingdom(text)
                    if candidate_data.get("transfers_used"):
                        data["transfers_used"] = candidate_data["transfers_used"]
                        break
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
            fallback_data, fallback_conf = extract_best_alliance_power(candidates)
            if alliance_data_complete(fallback_data):
                data, best_conf = fallback_data, fallback_conf
            else:
                structured_data, structured_conf = extract_alliance_power_structured(image)
                if alliance_data_complete(structured_data):
                    data, best_conf = structured_data, structured_conf
                else:
                    data, best_conf = fallback_data, fallback_conf
            tags = data["alliance_tags"]
            ctx = best_context(contexts=contexts, tags=tags)
            if ctx:
                overlap = len(tags & ctx.alliance_tags)
                ctx.combined_power_top3 = data["combined_power_top3"] or ctx.combined_power_top3
                ctx.top3_alliances = data["top3_alliances"] or ctx.top3_alliances
                ctx.alliance_comment = data.get("ocr_comment", "")
                ctx.alliance_tags |= tags
                ctx.alliance_ocr_conf = max(ctx.alliance_ocr_conf, best_conf)
                if overlap == 0 and tags:
                    if len(contexts) > 1:
                        ctx.used_fallback_match = True
                if overlap > 0:
                    ctx.used_fallback_match = False
            file_meta.append(
                {
                    "file": uploaded_file.name,
                    "type": screen_type,
                    "matched_kingdom": ctx.kingdom if ctx else "",
                    "ocr_confidence": f"{best_conf:.1f}",
                }
            )
        elif screen_type == "mystic_trial":
            fallback_rows, fallback_tags, fallback_conf = extract_best_mystic_rows(candidates)
            if len(fallback_rows) >= 3:
                parsed_rows = fallback_rows
                tags = fallback_tags
                best_conf = fallback_conf
            else:
                structured_rows, structured_tags, structured_conf = extract_mystic_rows_structured(image)
                parsed_rows = merge_mystic_rows(structured_rows, fallback_rows)
                tags = structured_tags | fallback_tags
                best_conf = max(structured_conf, fallback_conf)
            ctx = best_context(contexts=contexts, tags=tags)
            if ctx:
                overlap = len(tags & ctx.alliance_tags)
                ctx.alliance_tags |= tags
                if overlap == 0 and tags:
                    if len(contexts) > 1:
                        ctx.used_fallback_match = True
                if overlap > 0:
                    ctx.used_fallback_match = False
            for parsed in parsed_rows:
                manual_check = False
                reasons: list[str] = []
                if not ctx or not ctx.kingdom:
                    manual_check = True
                    reasons.append("no kingdom match")
                if ctx and ctx.used_fallback_match:
                    manual_check = True
                    reasons.append("fallback kingdom match")
                if not parsed["Alliance"] or parsed["Gamer tag"] == "*":
                    manual_check = True
                    reasons.append("missing alliance or gamer tag")
                if ctx and (not ctx.top3_alliances or not ctx.combined_power_top3):
                    manual_check = True
                    reasons.append("missing alliance power context")
                row_comment_parts = [parsed["OCR comment"]]
                if ctx and ctx.alliance_comment:
                    row_comment_parts.append(ctx.alliance_comment)
                row_comment = "; ".join(p for p in row_comment_parts if p)
                if row_comment:
                    manual_check = True
                    reasons.append("ocr uncertainty")

                all_rows.append(
                    {
                        "Kingdom": ctx.kingdom if ctx else "",
                        "Transfers used": ctx.transfers_used if ctx else "",
                        "Combined power of top 3 alliances": ctx.combined_power_top3 if ctx else "",
                        "Top 3 alliances": ctx.top3_alliances if ctx else "",
                        "Alliance": parsed["Alliance"],
                        "Mystic trial": parsed["Mystic trial"],
                        "Gamer tag": parsed["Gamer tag"],
                        "OCR comment": row_comment,
                        "Needs manual check": "yes" if manual_check else "no",
                        "Manual check reason": "; ".join(sorted(set(reasons))) or "none",
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
        "OCR comment",
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

if "is_extracting" not in st.session_state:
    st.session_state.is_extracting = False
if "extracted_df" not in st.session_state:
    st.session_state.extracted_df = None
if "file_meta" not in st.session_state:
    st.session_state.file_meta = None
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = {}

button_col, status_col = st.columns([1, 2])
with button_col:
    start_extract = st.button(
        "Extract data",
        type="primary",
        disabled=st.session_state.is_extracting,
    )
with status_col:
    if st.session_state.is_extracting:
        st.caption("Extraction in progress...")

if start_extract:
    st.session_state.is_extracting = True
    st.rerun()

if st.session_state.is_extracting:
    if not uploaded_files:
        st.session_state.is_extracting = False
        st.warning("Upload at least one screenshot.")
        st.stop()

    if not pytesseract:
        st.session_state.is_extracting = False
        st.error("`pytesseract` is not installed. Run `pip install -r requirements.txt` first.")
        st.stop()

    try:
        with st.spinner("Extracting data from screenshots..."):
            df, meta = run_extraction(uploaded_files)
            st.session_state.extracted_df = df
            st.session_state.file_meta = meta
            st.session_state.uploaded_image_bytes = {
                f.name: f.getvalue() for f in uploaded_files
            }
    except Exception as exc:
        st.session_state.is_extracting = False
        if pytesseract and isinstance(exc, pytesseract.TesseractNotFoundError):
            st.error(
                "Tesseract OCR was not found. Install Tesseract and set TESSERACT_CMD in environment if needed."
            )
            st.stop()
        raise
    finally:
        st.session_state.is_extracting = False

if st.session_state.extracted_df is not None:
    df = pd.DataFrame(st.session_state.extracted_df)
    meta = pd.DataFrame(st.session_state.file_meta or [])

    st.subheader("Detected files")
    st.dataframe(meta, use_container_width=True, hide_index=True)

    st.subheader("Extracted rows (editable)")
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        disabled=["Source file"],
    )
    st.session_state.extracted_df = edited_df

    source_options = sorted(edited_df["Source file"].dropna().unique().tolist()) if not edited_df.empty else []
    if source_options:
        st.subheader("Image review")
        selected_source = st.selectbox("Review source image", options=source_options)
        left_col, right_col = st.columns([1, 1.2])
        with left_col:
            image_bytes = st.session_state.uploaded_image_bytes.get(selected_source)
            if image_bytes:
                st.image(image_bytes, caption=selected_source, use_container_width=True)
        with right_col:
            review_df = edited_df[edited_df["Source file"] == selected_source]
            st.dataframe(review_df, use_container_width=True, hide_index=True)

    csv_data = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download extracted.csv",
        data=csv_data,
        file_name="extracted.csv",
        mime="text/csv",
    )
