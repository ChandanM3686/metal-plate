import streamlit as st
import streamlit.components.v1 as components
import os
import base64
import google.generativeai as genai
from PIL import Image
import io

# --- CONFIGURE GEMINI DIRECTLY ---
genai.configure(api_key="AIzaSyAJVC1UYqgv1DsPD52skz3y9n0YMG7UO2w")
model = genai.GenerativeModel("gemini-2.0-flash")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Metal Surface Text Extractor",
    page_icon="assets/icon.jpg",
    layout="wide"
)

# --- CUSTOM THEME STYLING ---
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
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #181818 0%, #222 100%);
            border-right: 2px solid rgba(218, 41, 28, 0.4);
        }
        h1 {
            text-align: center;
            font-weight: 800;
            font-size: 2.8rem;
            letter-spacing: 0.5px;
            padding-bottom: 0.3em;
            border-bottom: 2px solid #FFD700;
            display: inline-block;
            background: linear-gradient(90deg, #c8102e, #a60e28);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
            position: relative;
        }
        h1::after {
            content: "";
            position: absolute;
            left: 0; right: 0; bottom: -4px;
            height: 2px;
            background: linear-gradient(to right, #FFD700, #c8102e, #FFD700);
            opacity: 0.8;
            border-radius: 1px;
        }
        .hero-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 650;
            font-size: 3rem;
            text-align: center;
            text-transform: uppercase;
            margin: 2.2rem auto 1rem auto;
            background: linear-gradient(90deg, #ff1a1a, #e60000, #ff3333);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            -webkit-text-stroke: 0.6px #1a1a1a; 
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.35),
                         0 0 8px rgba(255, 50, 50, 0.7);
        }
        .hero-sub {
            color: #FFD700;
            text-align: center;
            font-weight: 500;
            font-size: 1.05rem;
            margin: 0.2rem 0 1.4rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="hero-title">Metal</div>
<div class="hero-sub">Ensure Quality. Detect with Precision. Powered by AI.</div>
""", unsafe_allow_html=True)

# --- SIDE-BY-SIDE LAYOUT ---
col1, col2 = st.columns([1, 1])

# --- LEFT COLUMN ---
with col1:
    st.markdown("""
    <style>
    .logo-text {
        font-family: 'Poppins', sans-serif;
        font-size: 5.1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
        margin: 4rem 0 1rem 0;
        padding-top: 2rem; 
        background: linear-gradient(90deg, #c8102e, #a60e28, #8b0d23);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        -webkit-text-stroke: 2px #FFD700;
        text-shadow: 
            0 2px 6px rgba(0, 0, 0, 0.6),
            0 0 12px rgba(255, 215, 0, 0.6),
            0 0 24px rgba(200, 16, 46, 0.8);
        animation: glowPulse 2s infinite alternate;
    }
    @keyframes glowPulse {
        from {
            text-shadow: 
                0 2px 6px rgba(0, 0, 0, 0.6),
                0 0 12px rgba(255, 215, 0, 0.5),
                0 0 24px rgba(200, 16, 46, 0.7);
        }
        to {
            text-shadow: 
                0 2px 6px rgba(0, 0, 0, 0.6),
                0 0 18px rgba(255, 215, 0, 0.9),
                0 0 36px rgba(200, 16, 46, 1);
        }
    }
    </style>
    <div class="logo-text">Metal Surface Text Extractor</div>
    """, unsafe_allow_html=True)

# --- RIGHT COLUMN ---
with col2:
    st.markdown("""
    <div style="text-align:center; font-size:2.1rem; font-weight:600; color:#fff; text-shadow:0 1px 3px rgba(0,0,0,0.4);">Upload / Capture Image</div>
    """, unsafe_allow_html=True)

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
            image_bytes = final_image.getvalue()
response = model.generate_content([
    {"mime_type": "image/jpeg", "data": image_bytes},
    {"text": """
                 Read and extract all text visible on the metal surface. 
Rules:
1) Read clockwise starting from the top-left/topmost text.
2) Prefer exact characters (letters, numbers, punctuation). Keep line breaks and approximate layout.
3) If any text is partially occluded or unreadable, mark it with [UNCLEAR:x] where x is the nearest readable chunk and optionally a guess in parentheses.
4) Return purely the extracted text only (no explanations).
"""}
])
result_text = response.text.strip()

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
                {result_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
