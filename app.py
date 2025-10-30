import streamlit as st
import streamlit.components.v1 as components
import os
import base64
import google.generativeai as genai
from PIL import Image
import io

# Image processing imports
import cv2
import numpy as np

# --- CONFIGURE GEMINI DIRECTLY ---
# Hardcoded key (you already set this earlier)
genai.configure(api_key="AIzaSyAJVC1UYqgv1DsPD52skz3y9n0YMG7UO2w")
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Utility: Preprocess Image for embossed metal OCR ---
def preprocess_image(file_bytes, target_max_dim=1600):
    """
    Preprocess an image for embossed/metal OCR:
    - Resize
    - Grayscale + bilateral filter (edge-preserving denoise)
    - CLAHE (local contrast enhancement)
    - Unsharp mask (sharpen)
    - Adaptive threshold + morphology to emphasize strokes
    - Deskew based on largest contour
    Returns JPEG bytes.
    """
    # decode bytes to cv2 image
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")

    # resize (preserve aspect)
    h, w = img.shape[:2]
    max_dim = max(h, w)
    scale = 1.0
    if max_dim > target_max_dim:
        scale = target_max_dim / max_dim
    elif max_dim < 500:
        scale = min(1.0, 500 / max_dim)
    if scale != 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # bilateral filter to remove noise but keep edges
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Unsharp mask (sharpen)
    gaussian = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

    # Adaptive threshold (binary-like to emphasize strokes)
    th = cv2.adaptiveThreshold(unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 12)

    # Morphology to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Deskew: rotate using minAreaRect of largest contour if meaningful
    contours, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 0.01 * (morph.shape[0]*morph.shape[1]):
            rect = cv2.minAreaRect(largest)
            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 1.0:
                (h2, w2) = img.shape[:2]
                center = (w2 // 2, h2 // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                # re-run a bit of processing after rotate
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
                gray = clahe.apply(gray)
                gaussian = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
                unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
                th = cv2.adaptiveThreshold(unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 31, 12)
                morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
                morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Combine sharpened color image with mask to emphasize characters visually (helps model)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(morph).convert("L")

    # Composite over white background using mask as alpha to highlight markings
    background = Image.new("RGB", pil_img.size, (255,255,255))
    background.paste(pil_img, mask=pil_mask)

    out_buf = io.BytesIO()
    background.save(out_buf, format="JPEG", quality=92)
    return out_buf.getvalue()


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Metal Surface Text Extractor",
    page_icon="assets/icon.jpg",
    layout="wide"
)

# --- CUSTOM THEME STYLING (kept minimal to match your UI) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500&display=swap');
        .stApp {
            background: url("https://static.vecteezy.com/system/resources/previews/008/555/699/non_2x/abstract-red-light-on-grey-black-cyber-geometric-circle-mesh-pattern-shadow-design-modern-futuristic-technology-background-vector.jpg") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Poppins', sans-serif;
            color: #E0E0E0;
        }
        #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="font-family:Poppins, sans-serif; font-weight:700; font-size:34px; text-align:center; color:#FFD700;">Metal Surface Text Extractor</div>
<div style="text-align:center; color:#FFD700; margin-bottom:18px;">Ensure Quality. Detect with Precision. Powered by AI.</div>
""", unsafe_allow_html=True)

# --- LAYOUT: Image upload / camera & analyze ---
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("""
    <div style="font-size:48px; font-weight:600; text-align:center; margin-top:30px; -webkit-text-stroke:0.6px #1a1a1a; color:transparent;
                 background:linear-gradient(90deg,#ff1a1a,#e60000,#ff3333); -webkit-background-clip:text;">Metal Surface Text Extractor</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""<div style="text-align:center; font-size:20px; color:#fff; margin-top:20px;">Upload / Capture Image</div>""", unsafe_allow_html=True)
    tab_gallery, tab_camera = st.tabs(["🖼️ Gallery", "📷 Camera"])

    picture = None
    uploaded_file = None

    with tab_gallery:
        file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg","bmp"], key="gallery_uploader")
        if file is not None:
            st.session_state["uploaded_image"] = file
        if "uploaded_image" in st.session_state:
            uploaded_file = st.session_state["uploaded_image"]

    with tab_camera:
        picture = st.camera_input("Take a photo", key="camera_input")

    final_image = picture if picture else uploaded_file

    if final_image:
        st.image(final_image, caption="Selected Inspection Card", use_container_width=True)

    analyze_button = st.button("Analyze")

    if analyze_button and final_image:
        with st.spinner("⚙️ Processing image..."):
            raw_bytes = final_image.getvalue()

            # 1) Try preprocessing (preferred)
            try:
                processed_bytes = preprocess_image(raw_bytes)
            except Exception as e:
                st.warning(f"Preprocessing failed, using raw image. ({e})")
                processed_bytes = raw_bytes

            # 2) Prepare a strict OCR prompt tuned for embossed metal text
            ocr_prompt = """
You are a precision OCR system specialized in reading embossed or engraved text on metallic and industrial surfaces.

Your task:
- Extract every alphanumeric character clearly visible on the metal surface.
- Handle glare, shadows, reflection, rotation, and partial visibility.
- Reconstruct continuous identifiers correctly — for example:
  • "C19 81" -> "C1981"
  • "JL3 W-4851-BB" -> "JL3W-4851-BB"
- Do NOT add spaces inside serial numbers, years, or part codes.
- If a character is slightly unclear, infer it intelligently from surrounding context rather than splitting it.
- Read text in clockwise order starting from the topmost region.
- Ignore bolts, scratches, background textures that look like letters.

Output:
- Return ONLY the extracted, cleaned text.
- No explanation, no commentary, no extra formatting.
- Each separate marking or code on a new line.
"""

            # 3) Send processed image to Gemini
            try:
                response = model.generate_content([
                    {"mime_type": "image/jpeg", "data": processed_bytes},
                    {"text": ocr_prompt}
                ])
                # `response.text` should contain the output for the model usage pattern used earlier
                result_text = getattr(response, "text", None)
                if not result_text:
                    # some SDK versions may return nested content; attempt other access patterns
                    try:
                        # try to access top-level 'candidates' or similar
                        result_text = str(response)
                    except Exception:
                        result_text = ""
                result_text = (result_text or "").strip()
            except Exception as e:
                # fallback: try original raw bytes
                st.warning(f"Model call on processed image failed, retrying with raw image. ({e})")
                try:
                    response = model.generate_content([
                        {"mime_type": "image/jpeg", "data": raw_bytes},
                        {"text": ocr_prompt}
                    ])
                    result_text = getattr(response, "text", "") or ""
                    result_text = result_text.strip()
                except Exception as ex:
                    st.error("Model call failed. Check API key and network.")
                    result_text = ""

        # 4) Display result (same style as before)
        st.markdown(f"""
        <div style="
            background: rgba(20, 20, 20, 0.85);
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 15px 20px;
            margin-top: 15px;
            box-shadow: 0 4px 12px rgba(255, 215, 0, 0.4);
            text-align: center;
        ">  
            <h3 style="color:#FFD700; text-align:center; margin-bottom:15px;">
                🛠️ Chassis Number
            </h3>
            <div style="color: #FFD700; font-size: 1.4rem; font-weight: bold; letter-spacing: 2px;">
                {st.session_state.setdefault('ocr_result', result_text)}
            </div>
        </div>
        """, unsafe_allow_html=True)
