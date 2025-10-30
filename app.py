import streamlit as st
import streamlit.components.v1 as components
import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import  Annotated
from pydantic import Field, BaseModel
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAJVC1UYqgv1DsPD52skz3y9n0YMG7UO2w"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


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

        /* Global app background & font */
        # .stApp {
        #     background: linear-gradient(135deg, #121212 0%, #1f1f1f 100%);
        #     color: #E0E0E0;
        #     font-family: 'Poppins', sans-serif;
        # }
        .stApp {
            background: url("https://static.vecteezy.com/system/resources/previews/008/555/699/non_2x/abstract-red-light-on-grey-black-cyber-geometric-circle-mesh-pattern-shadow-design-modern-futuristic-technology-background-vector.jpg") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Poppins', sans-serif;
            color: #E0E0E0;
        }

        #MainMenu {visibility: hidden;}

        /* Hide Streamlit footer */
        footer {visibility: hidden;}

        /* Hide Streamlit header (default top bar) */
        header {visibility: hidden;}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #181818 0%, #222 100%);
            border-right: 2px solid rgba(218, 41, 28, 0.4);
        }
        section[data-testid="stSidebar"] .css-1d391kg {
            color: #FFD700;
        }

        /* Main title */
        h1 {
        text-align: center;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: 0.5px;
        padding-bottom: 0.3em;
        border-bottom: 2px solid #FFD700; /* Gold underline */
        display: inline-block;

        /* Smooth metallic crimson gradient */
        background: linear-gradient(90deg, #c8102e, #a60e28);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        /* Soft glow */
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);

        /* Embossed effect */
        position: relative;
        }
        h1::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: -4px;
            height: 2px;
            background: linear-gradient(to right, #FFD700, #c8102e, #FFD700);
            opacity: 0.8;
            border-radius: 1px;
        }
        h1::before {
            content: attr(data-text);
            position: absolute;
            left: 2px;
            top: 2px;
            background: linear-gradient(90deg, #9b0e24, #7f0b1e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            z-index: -1;
            opacity: 0.4;
        }
            
        /* Subheadings */
        h2, h3, .stSubheader {
            color: #FFD700;
            text-align: center;
            font-weight: 500;
            margin-top: -8px;
            margin-bottom: 25px;
        }

        /* Hide the default red "Choose an image" button */
        /* Make "Choose an image" big and themed */
        .stFileUploader label {
            color: #FFD700 !important; /* Gold text */
            font-weight: 800;
            font-size: 1.6rem; /* Bigger size */
            text-align: center;
            display: block;
            margin-bottom: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            text-shadow: 0 0 8px rgba(255, 215, 0, 0.8), 0 0 14px rgba(255, 215, 0, 0.6);
        }


        /* File uploader container */
        .stFileUploader {
            background: rgba(20, 20, 20, 0.85);
            border: 2px solid rgba(255, 215, 0, 0.5);
            border-radius: 14px;
            padding: 25px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.6);
            transition: all 0.3s ease-in-out;
            text-align: center;
        }

        /* Hover effect on container */
        .stFileUploader:hover {
            border-color: #FFD700;
            box-shadow: 0 8px 24px rgba(255, 215, 0, 0.35);
            transform: translateY(-3px);
        }

        /* Dropzone styling */
        .stFileUploader div[data-testid="stFileUploaderDropzone"] {
            background: rgba(30, 30, 30, 0.95);
            border-radius: 10px;
            padding: 15px;
            border: 1.5px dashed rgba(255, 215, 0, 0.4);
            transition: border-color 0.3s ease-in-out;
        }

        .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #FFD700;
        }

        /* Browse files button inside uploader */
        .stFileUploader [data-testid="stBaseButton-secondary"] {
            background: linear-gradient(90deg, #c8102e, #a60e28);
            color: white !important; /* Softer warm white */
            font-weight: 500; /* Less bold */
            text-transform: capitalize;
            padding: 0.4em 1.4em;
            border-radius: 25px; /* Pill shape */
            border: none;
            box-shadow: 0 4px 10px rgba(200, 16, 46, 0.4);
            transition: all 0.3s ease;
        }

        .stFileUploader [data-testid="stBaseButton-secondary"]:hover {
            background: linear-gradient(90deg, #FFD700, #e6c200); /* gold hover */
            color: black !important;
            box-shadow: 0 6px 16px rgba(255, 215, 0, 0.6);
            transform: translateY(-2px);
        }

            
        /* Custom browse button */
        .stFileUploader button {
            background: linear-gradient(90deg, #c8102e, #a60e28);
            color: #fff8dc; /* Cornsilk-like warm white */
            font-weight: 500; /* Lighter than bold */
            border-radius: 8px;
            padding: 0.5em 1.2em;
            border: none;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 3px 10px rgba(200, 16, 46, 0.4);
        }

        .stFileUploader button:hover {
            background: linear-gradient(90deg, #FFD700, #e6c200);
            color: black !important;
            box-shadow: 0 6px 16px rgba(255, 215, 0, 0.4);
            transform: translateY(-2px);
        }

        

        
        /* Fake upload button (styled like browse) */
        .custom-upload-btn {
            display: inline-block;
            background: linear-gradient(90deg, #FFD700, #e6c200);
            color: black;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 3px 10px rgba(255, 215, 0, 0.4);
        }

        .custom-upload-btn:hover {
            background: linear-gradient(90deg, #c8102e, #a60e28);
            color: white;
            box-shadow: 0 6px 16px rgba(200, 16, 46, 0.4);
            transform: translateY(-2px);
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #c8102e, #a60e28); /* Smooth Honda crimson gradient */
            color: white;
            border-radius: 8px;
            padding: 0.65em 1.5em;
            border: none;
            font-weight: 700;
            font-size: 1rem;
            box-shadow: 0 4px 12px rgba(200, 16, 46, 0.4); /* soft crimson shadow */
            transition: all 0.3s ease-in-out;
        }

        /* Hover effect */
        .stButton>button:hover {
            background: linear-gradient(90deg, #FFD700, #e6c200); /* Gold metallic gradient */
            color: #000000;
            box-shadow: 0 6px 18px rgba(255, 215, 0, 0.6); /* Gold glow */
            transform: translateY(-2px); /* subtle lift effect */
        }

       .custom-file-upload {
            font-family: 'Poppins', sans-serif;
            font-weight: 400 !important; /* Force normal weight */
            font-style: normal;
            text-transform: capitalize;
            letter-spacing: 0.5px;
        }
            /* Tables */
        .stDataFrame, .stTable {
            border: 1px solid rgba(218, 41, 28, 0.4);
            border-radius: 8px;
            overflow: hidden;
        }
        .stDataFrame tbody tr:hover {
            background-color: rgba(255, 215, 0, 0.08);
        }

        /* Images */
        .stImage img {
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
        }
        .hero-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 650; /* smooth boldness */
            font-size: 3rem; /* a bit bigger, visible */
            text-align: center;
            text-transform: uppercase;
            margin: 2.2rem auto 1rem auto;

            /* Brighter Honda red metallic gradient */
            background: linear-gradient(90deg, #ff1a1a, #e60000, #ff3333);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;

            /* Stroke for clarity */
            -webkit-text-stroke: 0.6px #1a1a1a; 

            /* Subtle glow + smooth shadow for visibility */
            text-shadow: 
                0 2px 4px rgba(0, 0, 0, 0.35),
                0 0 8px rgba(255, 50, 50, 0.7);
        }




        .hero-title::after {
            content: "";
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            bottom: -10px;
            width: 70%;                  /* underline length relative to text */
            max-width: 480px;            /* cap for large screens */
            height: 3px;
            background: linear-gradient(90deg, transparent, #FFD700, transparent);
            border-radius: 2px;
        }

        .hero-sub {
            color: #FFD700;              /* Gold accent */
            text-align: center;
            font-weight: 500;
            font-size: 1.05rem;
            margin: 0.2rem 0 1.4rem 0;
        }
        
        .glow-logo {
            animation: glow 2s infinite alternate;
        }

        @keyframes glow {
            from {
                filter: drop-shadow(0 0 5px #FFD700) drop-shadow(0 0 10px #FFD700);
            }
            to {
                filter: drop-shadow(0 0 15px #FFD700) drop-shadow(0 0 25px #FFD700);
            }
        }


    </style>
""", unsafe_allow_html=True)



# --- HEADER ---
st.markdown("""
<div class="hero-title">Metal</div>
<div class="hero-sub">Ensure Quality. Detect with Precision. Powered by AI.</div>
""", unsafe_allow_html=True)

# --- SIDE-BY-SIDE LAYOUT ---
col1, col2 = st.columns([1, 1])  # 50-50 split

# --- LEFT COLUMN: Bike Image Placeholder ---
with col1:
   st.markdown("""
    <style>
    .logo-text {
        font-family: 'Poppins', sans-serif;
        font-size: 5.1rem;                  /* Big size */
        font-weight: 500;                 /* Super bold */
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
        margin: 4rem 0 1rem 0;            /* Top margin pushes it down */
        padding-top: 2rem; 

        /* Gradient metallic crimson */
        background: linear-gradient(90deg, #c8102e, #a60e28, #8b0d23);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        /* Stroke (outline) */
        -webkit-text-stroke: 2px #FFD700;   /* Gold outline */
        
        /* Shadow effects */
        text-shadow: 
            0 2px 6px rgba(0, 0, 0, 0.6),       /* dark base shadow */
            0 0 12px rgba(255, 215, 0, 0.6),    /* gold glow */
            0 0 24px rgba(200, 16, 46, 0.8);    /* crimson aura */
        
        /* Smooth edges */
        border-radius: 8px;
    }

    /* Animation for subtle glowing */
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

    .logo-text {
        animation: glowPulse 2s infinite alternate;
    }
    </style>

    <div class="logo-text">Metal Surface Text Extractor</div>
""", unsafe_allow_html=True)
# --- RIGHT COLUMN: Upload & Analyze ---
# --- RIGHT COLUMN: Upload & Analyze ---
with col2:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@600&display=swap');

    .upload-title {
        font-family: 'Nunito Sans', sans-serif;
        font-weight: 600;
        font-size: 2.1rem;
        text-align: center;
        margin: 1rem 0 0.9rem 0;
        letter-spacing: 0.6px;
        color: #ffffff;
        -webkit-text-stroke: 0.1px #111;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
    }

    /* Tabs styling */
    div[data-baseweb="tab-list"] {
        justify-content: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    button[data-baseweb="tab"] {
        background: rgba(20,20,20,0.85);
        color: #FFD700 !important;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        padding: 10px 20px;
        border: 1.5px solid rgba(255,215,0,0.4);
        transition: all 0.3s ease;
    }
    button[data-baseweb="tab"]:hover {
        background: linear-gradient(90deg, #c8102e, #a60e28);
        color: #fff !important;
        border-color: #FFD700;
        box-shadow: 0 4px 12px rgba(255,215,0,0.5);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #FFD700, #e6c200);
        color: black !important;
        border-color: #FFD700;
        box-shadow: 0 6px 16px rgba(255,215,0,0.6);
    }

    /* Camera input styling */
    div[data-testid="stCameraInput"] {
        background: rgba(20,20,20,0.85);
        border: 2px dashed rgba(255, 215, 0, 0.4);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    div[data-testid="stCameraInput"] label {
        color: #FFD700 !important;
        font-weight: 700;
        font-size: 1.2rem;
        text-shadow: 0 0 8px rgba(255,215,0,0.6);
    }
    </style>

    <div class="upload-title">Upload / Capture Image</div>
    """, unsafe_allow_html=True)
    

    tab_gallery, tab_camera = st.tabs(["🖼️ Gallery", "📷 Camera"])

    picture = None
    uploaded_file = None

    with tab_gallery:
        file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg","bmp"], key="gallery_uploader")

        # Store file in session state to prevent reset on mobile
        if file is not None:
            st.session_state["uploaded_image"] = file

        if "uploaded_image" in st.session_state:
            uploaded_file = st.session_state["uploaded_image"]

    with tab_camera:
        picture = st.camera_input("Take a photo", key="camera_input")

    # Decide which input to use
    final_image = picture if picture else uploaded_file

    if final_image:
        st.image(final_image, caption="Selected Inspection Card", use_container_width=True)

    analyze_button = st.button("Analyze")

    if analyze_button and final_image:
        with st.spinner("⚙️ Processing image..."):
            # Convert uploaded file (BytesIO) to base64
            encoded_image = base64.b64encode(final_image.getvalue()).decode("utf-8")

            # Create message for Gemini
            message = HumanMessage(content=[
                {"type": "text", "text": """
                 Read and extract all text visible on the metal surface. 
Rules:
1) Read clockwise starting from the top-left/topmost text.
2) Prefer exact characters (letters, numbers, punctuation). Keep line breaks and approximate layout.
3) If any text is partially occluded or unreadable, mark it with [UNCLEAR:x] where x is the nearest readable chunk and optionally a guess in parentheses.
4) Return purely the extracted text only (no explanations).
                """},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
            ])

            response = llm.invoke([message])
            result_text = response.content.strip()

        st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True
    )

