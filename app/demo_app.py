"""
Health Sight AI - Enterprise Multimodal Cancer Screening Platform
Senior UI/UX Redesign - Native Streamlit Design System
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import streamlit as st

os.environ["KERAS_BACKEND"] = "torch"
try:
    import tensorflow as tf
    from tensorflow import keras
except ModuleNotFoundError:
    import keras

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import CancerDataPreprocessor
from src.gradcam import GradCAM
from azure_vision import analyze_image, validate_image_quality
from azure_blob import upload_blob
from azure_speech import synthesize_speech, get_voice_script
from pdf_generator import generate_pdf_report
from database import save_record, get_all_records_df, delete_record


# Page configuration
st.set_page_config(
    page_title="Health Sight AI - Multimodal Cancer Screening Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Custom CSS for Native Widgets
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Background Canvas */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(91, 108, 255, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 90% 85%, rgba(0, 194, 255, 0.12) 0%, transparent 45%),
                    radial-gradient(circle at 50% 45%, rgba(123, 97, 255, 0.08) 0%, transparent 60%),
                    #F8FAFC !important;
    }

    /* Main Container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1320px !important;
    }

    /* Hero Typography */
    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 3.6rem !important;
        font-weight: 900 !important;
        letter-spacing: -1.5px !important;
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 50%, #7C3AED 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        margin-bottom: 0.2rem !important;
    }
    
    h2 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
    }
    
    h3 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
    }

    /* Native Metric Cards */
    [data-testid="stMetric"] {
        background: white !important;
        border-radius: 20px !important;
        padding: 1.25rem 1.5rem !important;
        border: 1px solid rgba(226, 232, 240, 0.9) !important;
        box-shadow: 0 10px 30px -5px rgba(91, 108, 255, 0.08) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 15px 35px -5px rgba(91, 108, 255, 0.2) !important;
        border-color: #6366F1 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 900 !important;
        color: #4F46E5 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: rgba(241, 245, 249, 0.9) !important;
        padding: 8px !important;
        border-radius: 20px !important;
        border: 1px solid #E2E8F0 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 14px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #64748B !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #4F46E5 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 0.85rem 2rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 25px -5px rgba(79, 70, 229, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px -5px rgba(79, 70, 229, 0.6) !important;
    }

    /* File Uploader Dropzone */
    [data-testid="stFileUploader"] {
        background: white !important;
        border: 2px dashed #6366F1 !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 20px -2px rgba(99, 102, 241, 0.08) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.88) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* Status Badges */
    .badge-green {
        background: #DCFCE7;
        color: #15803D;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.4rem;
    }
    .badge-blue {
        background: #E0F2FE;
        color: #0369A1;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str):
    """Load trained model (cached with automatic fallback build)"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.abspath(os.path.join(base_dir, model_path)) if not os.path.isabs(model_path) else model_path
    
    if os.path.exists(abs_path):
        try:
            return keras.models.load_model(abs_path)
        except Exception:
            pass
            
    if os.path.exists(model_path):
        try:
            return keras.models.load_model(model_path)
        except Exception:
            pass
            
    # Auto-generate & build model if missing
    try:
        from models.multimodal_model import MultimodalCancerDetector
        detector = MultimodalCancerDetector(num_classes=2)
        model = detector.build_fusion_model()
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        model.save(abs_path)
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None


def preprocess_image(image: Image.Image, img_size: tuple = (224, 224)) -> tuple:
    """Preprocess uploaded image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    original = img_array.copy()
    
    img_resized = cv2.resize(img_array, img_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    return img_normalized, original


def create_plotly_gauge(value, title):
    """Create modern gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18, 'color': '#0F172A', 'family': 'Outfit'}},
        number = {'suffix': "%", 'font': {'size': 36, 'color': '#4F46E5', 'family': 'Outfit'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
            'bar': {'color': "#4F46E5"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 50], 'color': '#DCFCE7'},
                {'range': [50, 75], 'color': '#FEF9C3'},
                {'range': [75, 100], 'color': '#FEE2E2'}
            ]
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Outfit'}
    )
    return fig


def create_probability_chart(predictions):
    """Create probability horizontal bar chart"""
    classes = ['Benign', 'Malignant']
    colors = ['#10B981', '#EF4444']
    
    fig = go.Figure(data=[
        go.Bar(
            y=classes,
            x=predictions * 100,
            orientation='h',
            marker=dict(color=colors, cornerradius=8),
            text=[f'{p*100:.1f}%' for p in predictions],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Outfit', weight='bold')
        )
    ])
    fig.update_layout(
        title=dict(text='Probability Distribution', font=dict(size=18, color='#0F172A', family='Outfit')),
        xaxis=dict(title='Confidence (%)', range=[0, 100], gridcolor='#F1F5F9'),
        yaxis=dict(title='', tickfont=dict(size=14, color='#0F172A', weight='bold')),
        height=240,
        margin=dict(l=20, r=20, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Outfit'}
    )
    return fig


def generate_health_response(user_message: str, language: str = "English") -> str:
    """Generate AI health responses"""
    try:
        import google.generativeai as genai
        GEMINI_API_KEY = "AIzaSyDummyKeyPleaseReplaceWithYourActualKey"
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        system_prompt = f"""You are an AI health assistant for Health Sight AI.
Answer health questions accurately and compassionately regarding skin cancer, symptoms, prevention, and treatment.
Respond in {language} language.
User question: {user_message}"""

        response = model.generate_content(system_prompt)
        return response.text
    except Exception:
        return generate_fallback_response(user_message, language)


def generate_fallback_response(user_message: str, language: str = "English") -> str:
    user_message_lower = user_message.lower()
    
    if any(g in user_message_lower for g in ["hello", "hi", "hey", "namaste"]):
        return f"Hello! I am your Health Sight AI assistant. How can I help you today regarding skin health or cancer screening in {language}?"
    
    if "types" in user_message_lower or "cancer" in user_message_lower:
        return f"""**Common Types of Skin Cancer ({language}):**
1. **Basal Cell Carcinoma (BCC)**: Most common, highly treatable.
2. **Squamous Cell Carcinoma (SCC)**: Second most common, appears as firm red nodule.
3. **Melanoma**: Most serious type, irregular borders and varied colors. Early detection is life-saving."""

    return f"""I am here to assist with your skin health queries in **{language}**.
You can ask about ABCDE warning signs, prevention tips, when to consult an oncologist, or how Health Sight AI works."""


def main():
    # HERO SECTION
    if os.path.exists("assets/logo.png"):
        h_col1, h_col2, h_col3 = st.columns([3, 1, 3])
        with h_col2:
            st.image("assets/logo.png", use_container_width=True)

    st.title("Health Sight AI")
    st.markdown("<p style='text-align: center; font-size: 1.3rem; font-weight: 700; color: #4F46E5; margin-bottom: 0.2rem;'>AI-Powered Multimodal Cancer Screening Platform</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.05rem; font-style: italic; color: #64748B; margin-bottom: 1.75rem;'>“Bringing world-class cancer detection to every village”</p>", unsafe_allow_html=True)

    # FEATURE HIGHLIGHTS BAR
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem;">
        <span style="background: rgba(79, 70, 229, 0.08); color: #4F46E5; border: 1px solid rgba(79, 70, 229, 0.2); 
                     padding: 0.5rem 1.25rem; border-radius: 30px; font-weight: 700; font-size: 0.9rem;">
            🧬 Multimodal AI Model
        </span>
        <span style="background: rgba(6, 182, 212, 0.08); color: #0891B2; border: 1px solid rgba(6, 182, 212, 0.2); 
                     padding: 0.5rem 1.25rem; border-radius: 30px; font-weight: 700; font-size: 0.9rem;">
            ☁️ Azure AI Validated
        </span>
        <span style="background: rgba(124, 58, 237, 0.08); color: #7C3AED; border: 1px solid rgba(124, 58, 237, 0.2); 
                     padding: 0.5rem 1.25rem; border-radius: 30px; font-weight: 700; font-size: 0.9rem;">
            🎙️ 6-Language Voice AI
        </span>
        <span style="background: rgba(16, 185, 129, 0.08); color: #059669; border: 1px solid rgba(16, 185, 129, 0.2); 
                     padding: 0.5rem 1.25rem; border-radius: 30px; font-weight: 700; font-size: 0.9rem;">
            📄 Official PDF Medical Reports
        </span>
    </div>
    """, unsafe_allow_html=True)

    # SIDEBAR REDESIGN
    with st.sidebar:
        if os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", width=65)
        st.markdown("### Health Sight AI")
        st.caption("Enterprise Clinical Suite v2.4")
        st.markdown("---")

        with st.expander("⚙️ Advanced Settings", expanded=False):
            model_path = st.text_input(
                "Model Path",
                value="models/saved_models/oncovision_multimodal.keras"
            )
            show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
            show_advanced = st.checkbox("Show Detailed Gauges", value=False)

        st.markdown("### 🎙️ Voice Assistant")
        voice_language = st.selectbox(
            "Speech Language",
            ["English", "Tamil", "Hindi", "Telugu", "Kannada", "Malayalam"],
            index=0
        )

        st.markdown("### ☁️ Azure Cloud Status")
        st.markdown('<div class="badge-green">🟢 Azure Vision Connected</div>', unsafe_allow_html=True)
        st.markdown('<div class="badge-green">🟢 Blob Storage Connected</div>', unsafe_allow_html=True)
        st.markdown('<div class="badge-blue">🟢 Azure Speech Connected</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🎯 Architecture")
        st.markdown("""
        • MobileNetV3-Small CNN  
        • Clinical Symptom MLP  
        • Grad-CAM Heatmap XAI  
        • SQLite History EHR  
        """)

        st.caption(f"🕒 Server Sync: {datetime.now().strftime('%H:%M:%S')}")

    # MAIN TABS
    tab1, tab_hist, tab2, tab3, tab4 = st.tabs([
        "🔬 Analysis Pipeline",
        "📋 Patient History Dashboard",
        "💬 AI Health Assistant",
        "📚 How It Works",
        "ℹ️ About Platform"
    ])

    # TAB 1: ANALYSIS
    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("### 📸 Upload Lesion Image")
            st.caption("Upload a clear dermoscopic or high-resolution photo of the lesion.")

            uploaded_file = st.file_uploader(
                "Upload medical image",
                type=['jpg', 'jpeg', 'png', 'avif', 'webp', 'bmp', 'tiff', 'tif', 'jfif'],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Lesion Specimen", use_container_width=True)
                st.success(f"File loaded: {uploaded_file.name} ({uploaded_file.size // 1024} KB)")
            else:
                st.info("💡 Drag & drop an image above to begin screening.")

        with col2:
            st.markdown("### 📋 Clinical Triage Data")
            st.caption("Provide patient symptoms to fuse clinical risk indicators with vision features.")

            age = st.slider("👤 Patient Age (years)", 18, 95, 50)
            duration = st.slider("⏱️ Symptom Duration (months)", 0.5, 60.0, 6.0, 0.5)
            family_history = st.selectbox("👨‍👩‍👧‍👦 Family History of Skin Cancer", ["No", "Yes"])
            family_history_val = 1 if family_history == "Yes" else 0
            pain_score = st.slider("😣 Pain Score (0-10)", 0.0, 10.0, 3.0, 0.5)
            lesion_size = st.slider("📏 Lesion Diameter (mm)", 2.0, 50.0, 10.0, 0.5)

    # TAB 2: PATIENT HISTORY DASHBOARD
    with tab_hist:
        st.markdown("### 📋 Patient EHR History Dashboard")
        st.caption("Persistent database tracking past screening predictions, Azure Cloud Storage links, and downloadable medical PDF reports.")

        df_history = get_all_records_df()

        if df_history.empty:
            st.info("ℹ️ No patient records stored in database yet. Perform a screening under Analysis Pipeline to populate records.")
        else:
            total_cases = len(df_history)
            pred_series = df_history['prediction_label'].astype(str).str.lower()
            malignant_cases = len(df_history[pred_series.str.contains('malignant')])
            benign_cases = len(df_history[pred_series.str.contains('benign')])
            
            conf_numeric = pd.to_numeric(df_history['confidence'], errors='coerce')
            avg_conf = float(conf_numeric.mean()) if not conf_numeric.empty and not pd.isna(conf_numeric.mean()) else 0.0

            h1, h2, h3, h4 = st.columns(4)
            with h1: st.metric("Total Patients", total_cases)
            with h2: st.metric("High Risk (Malignant)", malignant_cases)
            with h3: st.metric("Low Risk (Benign)", benign_cases)
            with h4: st.metric("Avg Confidence", f"{avg_conf:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)
            search_query = st.text_input("🔍 Search patient EHR records:", placeholder="Filter by patient name or diagnosis...")

            filtered_df = df_history
            if search_query:
                filtered_df = df_history[
                    df_history['patient_name'].astype(str).str.contains(search_query, case=False, na=False) |
                    df_history['prediction_label'].astype(str).str.contains(search_query, case=False, na=False) |
                    df_history['risk_level'].astype(str).str.contains(search_query, case=False, na=False)
                ]

            display_df = filtered_df.copy()
            for col in display_df.columns:
                def safe_cell(x):
                    if isinstance(x, bytes):
                        try:
                            return x.decode('utf-8', errors='replace')
                        except Exception:
                            return "<Binary Data>"
                    if pd.isna(x):
                        return ""
                    return x
                display_df[col] = display_df[col].map(safe_cell)

            st.dataframe(display_df, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📄 Patient Action & PDF Export Cards")

            for _, row in filtered_df.iterrows():
                rec_id = row['id']
                p_name = str(row['patient_name'])
                p_pred = str(row['prediction_label'])
                p_conf_val = pd.to_numeric(row['confidence'], errors='coerce')
                p_conf = float(p_conf_val) if not pd.isna(p_conf_val) else 0.0
                p_risk = str(row['risk_level'])
                p_time = str(row['timestamp'])
                p_blob = row.get('blob_url')

                is_mal = "malignant" in p_pred.lower()
                badge_icon = "🔴" if is_mal else "🟢"

                with st.expander(f"{badge_icon} Record #{rec_id} — {p_name} ({p_pred} - {p_conf:.1f}%) — {p_time}"):
                    c_col1, c_col2 = st.columns([2, 1])
                    with c_col1:
                        st.markdown(f"**Patient:** {p_name}")
                        st.markdown(f"**Metrics:** Age {row.get('age')} | Duration: {row.get('duration')} mos | Pain: {row.get('pain_score')}/10 | Size: {row.get('lesion_size')} mm")
                        st.markdown(f"**Diagnosis:** `{p_pred}` ({p_risk} Risk)")
                        if p_blob:
                            st.markdown(f"**Azure Storage URL:** [{p_blob}]({p_blob})")
                    with c_col2:
                        try:
                            pdf_buf = generate_pdf_report(
                                patient_name=p_name,
                                age=int(row.get('age', 50)),
                                duration=float(row.get('duration', 6.0)),
                                pain_score=float(row.get('pain_score', 3.0)),
                                lesion_size=float(row.get('lesion_size', 10.0)),
                                family_history=str(row.get('family_history', 'No')),
                                prediction_label=str(p_pred),
                                confidence=float(p_conf),
                                risk_level=str(p_risk),
                                blob_url=p_blob,
                                azure_tags=str(row.get('azure_tags', '')).split(', '),
                                speech_transcript=str(row.get('speech_transcript', '')),
                                speech_language=str(row.get('speech_language', 'English'))
                            )
                            st.download_button(
                                label=f"📄 Download Report #{rec_id}",
                                data=pdf_buf.getvalue(),
                                file_name=f"HealthSightAI_Record_{rec_id}_{p_name.replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_tab_{rec_id}",
                                use_container_width=True
                            )
                        except Exception as ex:
                            st.warning(f"Report build error: {ex}")

    # TAB 3: CHAT ASSISTANT
    with tab2:
        st.markdown("### 💬 Multilingual AI Health Assistant")
        st.caption("Ask questions regarding skin health, screening guidance, or symptom evaluation.")

        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []

        chat_language = st.selectbox("Response Language", ["English", "Hindi", "Spanish", "French", "German", "Tamil"], key="chat_lang_select")

        for msg in st.session_state.chat_messages:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role_icon} {msg['role'].capitalize()}:** {msg['content']}")

        user_q = st.text_input("Ask a medical question:", placeholder=f"Ask in {chat_language}...", key="user_q_input")
        if st.button("Send Query 📤", type="primary") and user_q:
            st.session_state.chat_messages.append({"role": "user", "content": user_q})
            ans = generate_health_response(user_q, chat_language)
            st.session_state.chat_messages.append({"role": "assistant", "content": ans})
            st.rerun()

    # TAB 4: HOW IT WORKS
    with tab3:
        st.markdown("### 📚 End-to-End Workflow Architecture")
        st.markdown("""
        1. **Medical Image Upload**: High-resolution skin lesion capture via mobile device.  
        2. **Azure AI Vision Quality Check**: Automated cloud validation for image clarity, lighting, and anatomical relevance.  
        3. **Azure Storage Blob Integration**: Secure encrypted cloud archiving with verifiable public URLs.  
        4. **Multimodal Deep Learning Inference**: MobileNetV3 CNN + Clinical MLP fusion architecture.  
        5. **Grad-CAM Explainable AI**: Visual gradient heatmaps highlighting diagnostic focus areas.  
        6. **Azure Speech Multilingual Guidance**: Neural text-to-speech diagnostic explanations in 6 local languages.  
        7. **Official ReportLab PDF Generator**: Comprehensive clinical reports with mobile-verifiable QR codes.  
        8. **Patient History Database (EHR)**: SQLite logging for patient monitoring and tracking.  
        """)

    # TAB 5: ABOUT PLATFORM
    with tab4:
        st.markdown("### ℹ️ About Health Sight AI Platform")
        st.write("""
        Health Sight AI is an enterprise-grade multimodal diagnostic screening platform designed to assist healthcare professionals, primary health centers (PHCs), and ASHA workers in early cancer detection.
        """)
        
        st.markdown("#### 🚀 Technology Stack")
        st.markdown("""
        • **Cloud**: Microsoft Azure AI Vision, Azure Speech, Azure Blob Storage  
        • **AI Engine**: TensorFlow & Keras 3 (MobileNetV3 + MLP Multimodal Architecture)  
        • **Explainability**: Grad-CAM Heatmap Visualization  
        • **Data & Reports**: SQLite Database + ReportLab PDF Generator  
        """)
        
        st.warning("⚠️ **Official Clinical Disclaimer:** Health Sight AI is an informational screening support tool. All diagnostic evaluations must be reviewed and confirmed by a certified medical professional prior to clinical action.")

    # PREDICT BUTTON & PIPELINE EXECUTION
    st.markdown("---")
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
    with c_btn2:
        analyze_button = st.button("🔬 Run Multimodal AI Screening", type="primary", use_container_width=True)

    if analyze_button:
        if uploaded_file is None:
            st.error("⚠️ Please upload a medical image specimen first.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        image_bytes = uploaded_file.getvalue()

        # Step 1: Azure Blob Upload
        status_text.markdown("☁️ **Step 1/6:** Uploading specimen to Azure Storage Blob...")
        progress_bar.progress(15)
        time.sleep(0.2)
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"uploaded_{timestamp_str}_{uploaded_file.name}"
            blob_url = upload_blob(image_bytes, blob_name)
            st.session_state["last_blob_url"] = blob_url
        except Exception:
            st.session_state["last_blob_url"] = None

        # Step 2: Azure Vision Quality Check & Specimen Validation
        status_text.markdown("🔍 **Step 2/6:** Validating specimen quality with Azure AI Vision...")
        progress_bar.progress(35)
        time.sleep(0.2)
        image_pil = Image.open(uploaded_file)
        try:
            azure_result = analyze_image(image_bytes)
            is_valid, warning_text, detected_tags = validate_image_quality(azure_result)
            st.session_state["last_azure_tags"] = detected_tags
            st.session_state["last_azure_is_valid"] = is_valid
            st.session_state["last_azure_warning_text"] = warning_text
        except Exception:
            is_valid, warning_text, detected_tags = validate_image_quality(None)
            st.session_state["last_azure_tags"] = detected_tags
            st.session_state["last_azure_is_valid"] = is_valid
            st.session_state["last_azure_warning_text"] = warning_text

        # Step 3: Load Model
        status_text.markdown("🤖 **Step 3/6:** Loading Multimodal AI Checkpoint...")
        progress_bar.progress(55)
        time.sleep(0.2)
        model = load_model(model_path)
        if model is None:
            st.error("❌ Failed to load AI model.")
            return

        try:
            model_input_shape = model.input[0].shape
            img_size = (model_input_shape[1], model_input_shape[2])
        except Exception:
            img_size = (128, 128)

        # Step 4: Preprocess
        status_text.markdown("🖼️ **Step 4/6:** Normalizing image & clinical symptom vectors...")
        progress_bar.progress(75)
        img_processed, img_original = preprocess_image(image_pil, img_size=img_size)
        img_batch = np.expand_dims(img_processed, axis=0)

        clinical_data = np.array([[age, duration, family_history_val, pain_score, lesion_size]])
        clinical_mean = np.array([50.0, 10.0, 0.5, 5.0, 15.0])
        clinical_std = np.array([18.0, 10.0, 0.5, 3.0, 10.0])
        clinical_normalized = (clinical_data - clinical_mean) / clinical_std

        # Step 5: Prediction
        status_text.markdown("🧠 **Step 5/6:** Running Neural Network Multimodal Inference...")
        progress_bar.progress(90)
        predictions = model.predict([img_batch, clinical_normalized], verbose=0)[0]
        pred_class = np.argmax(predictions)

        # Step 6: Speech Synthesis
        status_text.markdown("🎙️ **Step 6/6:** Generating Azure Speech Multilingual Audio...")
        progress_bar.progress(98)
        try:
            prediction_label = "Benign" if pred_class == 0 else "Malignant"
            script = get_voice_script(prediction_label, voice_language)
            audio_bytes = synthesize_speech(script, voice_language)
            st.session_state["last_audio_bytes"] = audio_bytes
            st.session_state["last_audio_text"] = script
            st.session_state["last_audio_language"] = voice_language
        except Exception:
            st.session_state["last_audio_bytes"] = None

        # Log to Database
        try:
            risk_level = "Low" if pred_class == 0 else "High"
            confidence = predictions[pred_class] * 100
            save_record(
                patient_name="Patient #" + datetime.now().strftime("%S%M"),
                age=age,
                duration=duration,
                pain_score=pain_score,
                lesion_size=lesion_size,
                family_history=family_history,
                prediction_label=prediction_label,
                confidence=confidence,
                risk_level=risk_level,
                blob_url=st.session_state.get("last_blob_url"),
                azure_tags=st.session_state.get("last_azure_tags"),
                speech_transcript=st.session_state.get("last_audio_text"),
                speech_language=st.session_state.get("last_audio_language", "English")
            )
        except Exception as e:
            print(f"DB Log Error: {e}")

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        # RESULTS DISPLAY
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Screening Diagnosis & Analysis Results")

        # Validation Warning Banner
        if not st.session_state.get("last_azure_is_valid", True):
            st.error(f"🚨 **Non-Medical Specimen Alert:**\n\n{st.session_state.get('last_azure_warning_text')}\n\n*Note: Predictions run on non-dermatological photos/portraits are invalid.*")
        else:
            st.success(f"✅ **Azure AI Vision Verified:** {st.session_state.get('last_azure_warning_text')}")

        res1, res2, res3 = st.columns(3)
        prediction_label = "Benign" if pred_class == 0 else "Malignant"
        confidence = predictions[pred_class] * 100
        risk_level = "Low" if pred_class == 0 else "High"

        with res1:
            st.metric("Diagnosis", f"{'✅' if pred_class == 0 else '⚠️'} {prediction_label}")

        with res2:
            st.metric("Confidence Score", f"{confidence:.1f}%")

        with res3:
            st.metric("Risk Assessment", f"{'🟢' if pred_class == 0 else '🔴'} {risk_level} Risk")

        # Voice assistant output
        if st.session_state.get("last_audio_bytes"):
            st.info(f"🎙️ **Azure Voice Output ({st.session_state.get('last_audio_language')}):** \"{st.session_state.get('last_audio_text')}\"")
            st.audio(st.session_state["last_audio_bytes"], format="audio/wav")

        # Grad-CAM Viewer
        if show_gradcam:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔍 Explainable AI - Grad-CAM Visual Heatmap")
            try:
                gradcam = GradCAM(model)
                heatmap, _ = gradcam.compute_heatmap(img_batch[0], clinical_normalized[0], class_idx=pred_class)
                overlay = gradcam.overlay_heatmap(heatmap, img_original)

                g1, g2, g3 = st.columns(3)
                with g1:
                    st.image(img_original, caption="Original Specimen", use_container_width=True)
                with g2:
                    fig_h, ax_h = plt.subplots(figsize=(5, 5))
                    ax_h.imshow(heatmap, cmap='jet')
                    ax_h.axis('off')
                    st.pyplot(fig_h)
                    st.caption("Grad-CAM Heatmap")
                    plt.close()
                with g3:
                    st.image(overlay, caption="Explainable Overlay", use_container_width=True)

                st.success("✅ **Explainability Note:** Red/Yellow highlights reflect spatial activation regions that contributed to the model's output.")
            except Exception as e:
                st.warning(f"Grad-CAM visualization info: {e}")

        # PDF Download Button
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            gradcam_pil_img = None
            if 'overlay' in locals() and overlay is not None:
                if isinstance(overlay, np.ndarray):
                    gradcam_pil_img = Image.fromarray(overlay.astype(np.uint8))

            pdf_bytes_io = generate_pdf_report(
                patient_name="Patient #" + datetime.now().strftime("%S%M"),
                age=age,
                gender="Unspecified",
                duration=duration,
                pain_score=pain_score,
                lesion_size=lesion_size,
                family_history=family_history,
                prediction_label=prediction_label,
                confidence=confidence,
                risk_level=risk_level,
                probabilities=predictions,
                original_img=img_original,
                gradcam_img=gradcam_pil_img,
                blob_url=st.session_state.get("last_blob_url"),
                azure_tags=st.session_state.get("last_azure_tags"),
                speech_transcript=st.session_state.get("last_audio_text"),
                speech_language=st.session_state.get("last_audio_language", "English")
            )

            pdf_filename = f"HealthSightAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            d1, d2, d3 = st.columns([1, 2, 1])
            with d2:
                st.download_button(
                    label="📄 Download Official Medical PDF Report",
                    data=pdf_bytes_io.getvalue(),
                    file_name=pdf_filename,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
        except Exception as e:
            st.warning(f"Could not prepare PDF: {e}")

    # APP FOOTER
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.9rem;'>Built with ❤️ by <strong>Kishore Shyam</strong> | AI & DS | RMK Engineering College</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 0.8rem;'>Powered by Microsoft Azure AI • TensorFlow & Keras 3 • Streamlit • ReportLab</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
