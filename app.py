import streamlit as st
import cv2
import numpy as np
import easyocr
import torch
import json
import time
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Text Extractor", layout="wide")
st.title("Document Text Extractor")
st.caption("English · OpenCV preprocessing → CRAFT (detection) + CRNN (recognition) via EasyOCR")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("Preprocessing")
enable_denoise  = st.sidebar.checkbox("Denoise",               value=True)
enable_deskew   = st.sidebar.checkbox("Deskew",                value=True)
enable_binarize = st.sidebar.checkbox("Binarize",              value=True)
binarize_method = st.sidebar.radio(
    "Binarize method", ["Otsu", "Adaptive"], horizontal=True,
    help="Otsu: good for clean docs. Adaptive: better for uneven lighting."
)
enable_morph   = st.sidebar.checkbox("Morphological cleanup",  value=False,
    help="Opens then closes small noise artifacts. Useful for scanned docs.")
enable_sharpen = st.sidebar.checkbox("Edge enhancement",       value=False,
    help="Unsharp mask — improves faint or blurry text.")

st.sidebar.markdown("---")
st.sidebar.subheader("EasyOCR Detection")
conf_thresh = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, 0.3, 0.05,
    help="Lower = catches more text but may include noise. Higher = stricter."
)
show_boxes = st.sidebar.checkbox("Show bounding boxes", value=True)

# ── Paragraph mode: sorts detected regions top-to-bottom, left-to-right
paragraph_mode = st.sidebar.checkbox(
    "Paragraph sort (reading order)", value=True,
    help="Sorts detected text regions into natural reading order."
)

st.sidebar.markdown("---")
device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
st.sidebar.markdown(f"**Device:** {device_label}")
st.sidebar.caption("EasyOCR uses PyTorch — GPU detected automatically.")

# ── CV Preprocessing Pipeline ─────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Sequential OpenCV preprocessing pipeline.
    Each step is independently togglable from the sidebar.
    Returns (processed_gray, step_info_dict).
    """
    info = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Denoise ──────────────────────────────────────────────────────────────
    if enable_denoise:
        noise_level = float(gray.std())
        if noise_level > 40:
            # Heavy noise → Non-Local Means (DL-friendly, slower)
            gray = cv2.fastNlMeansDenoising(gray, h=10,
                                             templateWindowSize=7,
                                             searchWindowSize=21)
            info["Denoise"] = f"NL-Means (noise σ={noise_level:.1f})"
        else:
            # Light noise → fast Gaussian
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            info["Denoise"] = f"Gaussian (noise σ={noise_level:.1f})"

    # ── Deskew ───────────────────────────────────────────────────────────────
    if enable_deskew:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=80,
            minLineLength=gray.shape[1] // 4,
            maxLineGap=20,
        )
        angle = 0.0
        if lines is not None:
            angles = [
                np.degrees(np.arctan2(y2 - y1, x2 - x1))
                for x1, y1, x2, y2 in lines[:, 0]
                if x2 != x1
            ]
            if angles:
                angle = float(np.median(angles))

        if 0.5 < abs(angle) < 45:
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(
                gray, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            info["Deskew"] = f"Corrected {angle:.2f}°"
        else:
            info["Deskew"] = "No skew detected"

    # ── Edge Enhancement (Unsharp Mask) ──────────────────────────────────────
    if enable_sharpen:
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        info["Edge Enhancement"] = "Unsharp mask"

    # ── Binarize ─────────────────────────────────────────────────────────────
    if enable_binarize:
        if binarize_method == "Otsu":
            _, gray = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=31, C=10,
            )
        info["Binarize"] = f"{binarize_method}"

    # ── Morphological Cleanup ─────────────────────────────────────────────────
    if enable_morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN,  kernel, iterations=1)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
        info["Morph Cleanup"] = "Open + Close"

    return gray, info


# ── EasyOCR ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading EasyOCR (CRAFT + CRNN) model…")
def load_reader() -> easyocr.Reader:
    """
    Loads EasyOCR once and caches it for the session.
    Internally uses:
      - CRAFT (CNN): character-region text detector
      - CRNN (CNN + BiLSTM + CTC): sequence recogniser
    GPU is used automatically if CUDA is available.
    """
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available())


def sort_reading_order(results: list) -> list:
    """
    Sorts EasyOCR bounding boxes into natural reading order
    (top-to-bottom, then left-to-right within the same line band).
    """
    def bbox_top(r):
        return min(pt[1] for pt in r[0])

    def bbox_left(r):
        return min(pt[0] for pt in r[0])

    # Group into line bands (within 15px vertical proximity)
    results_sorted = sorted(results, key=bbox_top)
    lines, current_line = [], []
    prev_top = None

    for r in results_sorted:
        top = bbox_top(r)
        if prev_top is None or abs(top - prev_top) < 15:
            current_line.append(r)
        else:
            lines.append(sorted(current_line, key=bbox_left))
            current_line = [r]
        prev_top = top

    if current_line:
        lines.append(sorted(current_line, key=bbox_left))

    return [r for line in lines for r in line]


def run_ocr(gray: np.ndarray) -> tuple[list, str]:
    """
    Runs CRAFT detection + CRNN recognition on the preprocessed image.
    Filters by confidence threshold and optionally sorts into reading order.
    """
    reader = load_reader()
    raw = reader.readtext(gray, detail=1)
    filtered = [r for r in raw if r[2] >= conf_thresh]

    if paragraph_mode:
        filtered = sort_reading_order(filtered)

    text = "\n".join(r[1] for r in filtered)
    return filtered, text


# ── Bounding Box Overlay ──────────────────────────────────────────────────────

def draw_boxes(img_bgr: np.ndarray, results: list) -> np.ndarray:
    vis = img_bgr.copy()
    for i, (bbox, text, conf) in enumerate(results):
        pts = np.array(bbox, dtype=np.int32)
        # Color shifts green→yellow with lower confidence
        green = int(200 * conf)
        red   = int(200 * (1 - conf))
        color = (0, green, red)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        x, y = pts[0]
        label = f"{text[:20]} ({conf:.2f})"
        cv2.putText(vis, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
    return vis


# ── Text Metrics ──────────────────────────────────────────────────────────────

def metrics(text: str) -> str:
    words = len(text.split())
    lines = len([l for l in text.splitlines() if l.strip()])
    chars = len(text.replace(" ", "").replace("\n", ""))
    return f"**Words:** {words} · **Lines:** {lines} · **Chars (non-space):** {chars}"


# ── Main: Upload & Process ────────────────────────────────────────────────────

files = st.file_uploader(
    "Upload image(s) — JPG, PNG, BMP, TIFF",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
    accept_multiple_files=True,
)

if not files:
    st.info("Upload one or more document images to begin.")
    st.stop()

all_results = {}

for idx, f in enumerate(files):
    st.divider()
    st.subheader(f.name)

    raw_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    img_bgr   = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not decode image. Please upload a valid image file.")
        continue

    # ── Preprocessing ────────────────────────────────────────────────────────
    t0 = time.time()
    gray, prep_info = preprocess(img_bgr)
    prep_ms = (time.time() - t0) * 1000

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown("**Preprocessed**")
        st.image(gray, clamp=True, use_container_width=True)

    with st.expander("Preprocessing steps applied", expanded=False):
        for k, v in prep_info.items():
            st.markdown(f"- **{k}:** {v}")
        st.caption(f"Total preprocessing time: {prep_ms:.1f} ms")

    # ── OCR ──────────────────────────────────────────────────────────────────
    with st.spinner("Running CRAFT + CRNN recognition…"):
        t1 = time.time()
        results, extracted_text = run_ocr(gray)
        ocr_ms = (time.time() - t1) * 1000

    # ── Bounding Box Overlay ─────────────────────────────────────────────────
    if show_boxes and results:
        st.markdown("**Detected Text Regions**")
        st.image(
            cv2.cvtColor(draw_boxes(img_bgr, results), cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )
        st.caption("Box color: green = high confidence → yellow = lower confidence")

    # ── Extracted Text Output ────────────────────────────────────────────────
    st.markdown(
        f"**Extracted Text** · {ocr_ms:.0f} ms · "
        f"{len(results)} regions detected · conf ≥ {conf_thresh}"
    )
    st.code(extracted_text, language=None)
    st.markdown(metrics(extracted_text))

    all_results[f.name] = extracted_text

    st.download_button(
        label=f"Download extracted text — {f.name}",
        data=extracted_text,
        file_name=Path(f.name).stem + "_extracted.txt",
        mime="text/plain",
        key=f"dl_{idx}_{f.name}",
    )

# ── Batch Export ──────────────────────────────────────────────────────────────
if len(all_results) > 1:
    st.divider()
    st.subheader("Batch Export")
    combined = "\n\n".join(
        f"=== {name} ===\n{txt}" for name, txt in all_results.items()
    )
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download all as .txt", combined,
            "batch_extracted.txt", "text/plain",
        )
    with c2:
        st.download_button(
            "Download all as .json",
            json.dumps(all_results, indent=2),
            "batch_extracted.json", "application/json",
        )