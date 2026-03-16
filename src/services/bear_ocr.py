import os
import re
from pathlib import Path
from collections import Counter

from PIL import Image, ImageOps, ImageFilter
import pytesseract

from config.settings import TESSERACT_CMD


if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def number_to_int(value: str) -> int:
    try:
        return int(re.sub(r"[^0-9]", "", value))
    except ValueError:
        return 0


def preprocess_for_ocr(image: Image.Image, scale: int = 2) -> Image.Image:
    base = ImageOps.exif_transpose(image).convert("RGB")
    gray = ImageOps.grayscale(base)
    contrasted = ImageOps.autocontrast(gray)
    if scale != 1:
        return contrasted.resize((contrasted.width * scale, contrasted.height * scale), Image.Resampling.LANCZOS)
    return contrasted


def looks_like_bear_screenshot(image: Image.Image) -> bool:
    width, height = image.size
    crops = [
        image.crop((int(width * 0.08), int(height * 0.08), int(width * 0.92), int(height * 0.24))),
        image.crop((int(width * 0.08), int(height * 0.16), int(width * 0.92), int(height * 0.34))),
    ]
    patterns = [
        r"\btrap\s*[12]\b",
        r"\bdamage\s+ranking\b",
        r"\bpersonal\s+damage\s+rewards?\b",
        r"\bdamage\s+points\b",
    ]

    for crop in crops:
        processed = preprocess_for_ocr(crop, scale=2)
        text = pytesseract.image_to_string(processed, config="--psm 6").strip().lower()
        normalized = re.sub(r"\s+", " ", text)
        if any(re.search(pattern, normalized) for pattern in patterns):
            return True
    return False


def detect_trap_type(image: Image.Image) -> str:
    width, height = image.size
    crop_bands = [
        (0.08, 0.20),
        (0.10, 0.22),
        (0.12, 0.24),
        (0.18, 0.28),
    ]

    votes = Counter()
    for top_ratio, bottom_ratio in crop_bands:
        title_crop = image.crop((int(width * 0.08), int(height * top_ratio), int(width * 0.92), int(height * bottom_ratio)))
        for processed in _title_preprocessing_variants(title_crop):
            text = pytesseract.image_to_string(processed, config="--psm 6").strip()
            detected = _trap_type_from_text(text)
            if detected != "Unknown":
                if _is_confident_trap_read(text, detected):
                    return detected
                votes[detected] += 1
                other = "Trap 1" if detected == "Trap 2" else "Trap 2"
                if votes[detected] >= 2 and votes[other] == 0:
                    return detected

    if not votes:
        return "Unknown"
    return votes.most_common(1)[0][0]


def _trap_type_from_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    compact = re.sub(r"\s+", "", normalized)

    if re.search(r"(trap|jtrap|jirap|lirap|ilirap|llirap|irirap)[^A-Za-z0-9]{0,4}[2Zz}]", compact, re.IGNORECASE):
        return "Trap 2"
    if re.search(r"(trap|jtrap|jirap|lirap|ilirap|llirap|irirap)[^A-Za-z0-9]{0,4}[1Il!|\])]", compact, re.IGNORECASE):
        return "Trap 1"
    return "Unknown"


def _is_confident_trap_read(text: str, detected: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    digit = "1" if detected == "Trap 1" else "2"
    if re.search(rf"\btrap\s*{digit}\b", normalized):
        return True
    if re.search(rf"\btrap\s*{digit}\b.*\bdamage\b.*\brewards?\b", normalized):
        return True
    return False


def _title_preprocessing_variants(image: Image.Image):
    base = preprocess_for_ocr(image, scale=3)
    yield base

    sharpened = base.filter(ImageFilter.SHARPEN)
    yield sharpened

    threshold = sharpened.point(lambda pixel: 255 if pixel > 170 else 0)
    yield threshold


def extract_bear_scores(image_path: str):
    image = Image.open(image_path)
    if not looks_like_bear_screenshot(image):
        return {"trap": "Excluded", "scores": [], "excluded": True}

    width, height = image.size
    trap_type = detect_trap_type(image)

    ranking_area = image.crop((0, int(height * 0.25), width, int(height * 0.95)))
    processed = preprocess_for_ocr(ranking_area, scale=3)
    data = pytesseract.image_to_data(processed, config="--psm 6", output_type=pytesseract.Output.DICT)

    damage_anchors = []
    for index, text in enumerate(data["text"]):
        text = text.strip().lower()
        if ("damage" in text or "points" in text) and int(data["conf"][index]) > 20:
            damage_anchors.append(index)

    rows_indices = []
    if damage_anchors:
        damage_anchors.sort(key=lambda idx: data["top"][idx])
        current_row = [damage_anchors[0]]
        for index in range(1, len(damage_anchors)):
            if abs(data["top"][damage_anchors[index]] - data["top"][current_row[0]]) < 60:
                current_row.append(damage_anchors[index])
            else:
                rows_indices.append(current_row)
                current_row = [damage_anchors[index]]
        rows_indices.append(current_row)

    results = []
    n_boxes = len(data["text"])
    for row_idx_list in rows_indices:
        avg_top = sum(data["top"][idx] for idx in row_idx_list) / len(row_idx_list)
        user_words = []
        damage_words = []

        for index in range(n_boxes):
            v_dist = data["top"][index] - avg_top
            if abs(v_dist) < 70:
                damage_words.append({"text": data["text"][index], "left": data["left"][index]})
            elif -250 < v_dist < -40:
                user_words.append({"text": data["text"][index], "left": data["left"][index], "top": data["top"][index]})

        user_words.sort(key=lambda item: (round(item["top"] / 30), item["left"]))
        damage_words.sort(key=lambda item: item["left"])

        user_line = " ".join(word["text"] for word in user_words)
        damage_line = " ".join(word["text"] for word in damage_words)
        score = _extract_score(damage_line, user_line)
        user = _clean_user(user_line)

        if "damage" in user.lower() and score == 0:
            continue
        if user.lower() == "demon finch" and score == 0:
            continue
        if any(value in user.lower() for value in ["points", "damage", "rewards"]):
            continue
        if score == 0:
            continue

        results.append({"username": user, "damage": score})

    return {"trap": trap_type, "scores": results, "excluded": False}


def _extract_score(damage_line: str, user_line: str) -> int:
    score_pattern = r"(?:Dam|Dan|Darn|Poi|Pai|Pni)\w*[:\s]*([0-9,.]+)"
    score_match = re.search(score_pattern, damage_line, re.IGNORECASE)
    if score_match:
        return number_to_int(score_match.group(1).strip())

    for num_str in re.findall(r"[0-9,.]+", f"{damage_line} {user_line}"):
        clean_num = number_to_int(num_str)
        if clean_num > 1_000_000:
            return clean_num
    return 0


def _clean_user(user_line: str) -> str:
    nom_pattern = r"[|{\[\(]\s*[Nn0][Oo][Mm]\s*[|\])}]"
    nom_matches = list(re.finditer(nom_pattern, user_line, re.IGNORECASE))
    if nom_matches:
        user = user_line[nom_matches[-1].end():].strip()
    else:
        nom_prefix_match = re.search(r"[Nn0][Oo][Mm][I|!|l|J]\s*(.*)", user_line, re.IGNORECASE)
        user = nom_prefix_match.group(1).strip() if nom_prefix_match else user_line.strip()

    user = re.sub(r"^[0-9Il|!>Â»*Â©Â®#@$Â§=\-\+~\s\(\)\{\}\.,_]+", "", user).strip()
    user = re.sub(_noise_pattern(_leading_noise()), "", user, flags=re.IGNORECASE).strip()
    user = re.sub(_noise_pattern(_trailing_noise(), suffix=True), "", user, flags=re.IGNORECASE).strip()
    user = re.sub(r"[\*Â©Â®Â§|]$", "", user).strip()
    user = re.sub(r"\s+[|]$", "", user).strip()
    user = re.sub(r"\s+\d+$", "", user).strip()
    user = re.sub(r"\s+[a-z]$", "", user, flags=re.IGNORECASE).strip()
    return user


def _noise_pattern(values: list[str], suffix: bool = False) -> str:
    joined = "|".join(re.escape(value) for value in values)
    if suffix:
        return rf"\s+(?:{joined})\s*$"
    return rf"^(?:{joined})\s+"


def _leading_noise() -> list[str]:
    return [
        "we", "P", "fa", "m", "ey", "ia", "hi", "ron", "ih", "sf", "Tom", "fd",
        "ee", "Ge", "D", "ya", "by", "a", "Ve", "va", "B", "f", "be", "et",
        "iT", "me", "nah", "Soe", "omm", "ear", "DI", "Dy", "oF", "D5", "pt", "a3", "Le", "Ko i",
    ]


def _trailing_noise() -> list[str]:
    return [
        "we", "P", "fa", "m", "ey", "ia", "hi", "ron", "ih", "sf", "Tom", "fd",
        "ee", "Ge", "D", "ya", "by", "a", "Ve", "va", "B", "f", "be", "et",
        "iT", "me", "nah", "Soe", "omm", "ear", "DI", "Dy", "oF", "D5", "pt", "a3", "Le",
        "re", "mae", "ay", "ay)", "ew",
    ]


if __name__ == "__main__":
    bear_dir = Path("img") / "bear"
    for filename in sorted(bear_dir.iterdir(), key=lambda path: int(path.stem) if path.stem.isdigit() else 999):
        if filename.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
            continue
        result = extract_bear_scores(str(filename))
        print(f"File: {filename.name} | Trap: {result['trap']}")
        for rank, score in enumerate(result["scores"], start=1):
            print(f"  {rank}. {score['username']}, {score['damage']}")
        print("-" * 20)
