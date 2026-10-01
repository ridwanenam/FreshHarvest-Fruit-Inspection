import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==============================================================================
# 1. Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="FreshHarvest - Fruit Quality Inspection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Clean & Direct)
st.markdown("""
<style>
    .main-header {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }
    .status-fresh {
        background-color: #059669;
        color: #ffffff;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
        font-size: 13px;
    }
    .status-spoiled {
        background-color: #dc2626;
        color: #ffffff;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
        font-size: 13px;
    }
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .kpi-number {
        font-size: 22px;
        font-weight: 700;
        color: #38bdf8;
    }
    .kpi-label {
        font-size: 12px;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. Target Classes
# ==============================================================================
CLASS_NAMES = [
    'F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry', 'F_Tamarillo', 'F_Tomato',
    'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango', 'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato'
]

CLASS_DISPLAY = {
    'F_Banana': {'fruit': 'Banana', 'status': 'Fresh', 'grade': 'Good Quality'},
    'F_Lemon': {'fruit': 'Lemon', 'status': 'Fresh', 'grade': 'Good Quality'},
    'F_Lulo': {'fruit': 'Lulo (Naranjilla)', 'status': 'Fresh', 'grade': 'Good Quality'},
    'F_Mango': {'fruit': 'Mango', 'status': 'Fresh', 'grade': 'Good Quality'},
    'F_Orange': {'fruit': 'Orange', 'status': 'Fresh', 'grade': 'Good Quality'},
    'F_Strawberry': {'fruit': 'Strawberry', 'status': 'Fresh', 'grade': 'Good Quality'},
    'F_Tamarillo': {'fruit': 'Tamarillo', 'status': 'Fresh', 'grade': 'Good Quality'},
    'F_Tomato': {'fruit': 'Tomato', 'status': 'Fresh', 'grade': 'Good Quality'},
    'S_Banana': {'fruit': 'Banana', 'status': 'Spoiled', 'grade': 'Bad Quality'},
    'S_Lemon': {'fruit': 'Lemon', 'status': 'Spoiled', 'grade': 'Bad Quality'},
    'S_Lulo': {'fruit': 'Lulo (Naranjilla)', 'status': 'Spoiled', 'grade': 'Bad Quality'},
    'S_Mango': {'fruit': 'Mango', 'status': 'Spoiled', 'grade': 'Bad Quality'},
    'S_Orange': {'fruit': 'Orange', 'status': 'Spoiled', 'grade': 'Bad Quality'},
    'S_Strawberry': {'fruit': 'Strawberry', 'status': 'Spoiled', 'grade': 'Bad Quality'},
    'S_Tamarillo': {'fruit': 'Tamarillo', 'status': 'Spoiled', 'grade': 'Bad Quality'},
    'S_Tomato': {'fruit': 'Tomato', 'status': 'Spoiled', 'grade': 'Bad Quality'}
}

# ==============================================================================
# 3. Model Loader with Release Download
# ==============================================================================
GITHUB_RELEASE_URL = "https://github.com/ridwanenam/FreshHarvest-Fruit-Inspection/releases/download/v1.0.0/resnet50_freshharvest_transfer_model.keras"

def download_model_from_release(target_path="models/resnet50_freshharvest_transfer_model.keras", url=GITHUB_RELEASE_URL):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        import requests
        with st.spinner("Downloading ResNet50 model weights from GitHub Release..."):
            response = requests.get(url, stream=True, timeout=120)
            if response.status_code == 200:
                with open(target_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                return True, "Download successful."
            else:
                return False, f"HTTP {response.status_code}: Model file not found in Release v1.0.0."
    except Exception as e:
        return False, str(e)

@st.cache_resource(show_spinner="Loading model...")
def load_model_pipeline():
    candidate_paths = [
        "models/resnet50_freshharvest_transfer_model.keras",
        "../week_6/resnet50_freshharvest_transfer_model.keras",
        "resnet50_freshharvest_transfer_model.keras",
        "../models/resnet50_freshharvest_transfer_model.keras"
    ]
    model_path = None
    for p in candidate_paths:
        if os.path.exists(p) and os.path.getsize(p) > 1000000:
            model_path = p
            break
            
    if not model_path:
        target_path = "models/resnet50_freshharvest_transfer_model.keras"
        success, dl_msg = download_model_from_release(target_path)
        if success:
            model_path = target_path
        else:
            return None, (
                f"Model file not found locally and could not be downloaded ({dl_msg}). "
                "Make sure you created Release v1.0.0 on GitHub with file "
                "'resnet50_freshharvest_transfer_model.keras'."
            )
        
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"preprocess_input": preprocess_input},
            safe_mode=False
        )
        return model, None
    except Exception as e:
        return None, str(e)

# ==============================================================================
# 4. Sidebar
# ==============================================================================
with st.sidebar:
    st.markdown("### FreshHarvest Logistics")
    st.markdown("Fruit Quality Inspection System")
    st.markdown("---")
    
    st.markdown("#### Performance Metrics")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Test Accuracy</div>
            <div class="kpi-number" style="color:#10b981;">98.68%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_sb2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Sorting Accuracy</div>
            <div class="kpi-number" style="color:#38bdf8;">99.05%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Model Architecture</div>
        <div style="font-size:14px; font-weight:600; color:#f8fafc; margin-top:2px;">ResNet50 (Transfer Learning)</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Supported Fruits (16 Classes)")
    st.caption("8 Fruits in Fresh and Spoiled condition:")
    produce_list = ["Banana", "Lemon", "Lulo", "Mango", "Orange", "Strawberry", "Tamarillo", "Tomato"]
    st.write(", ".join(produce_list))

# ==============================================================================
# 5. Header
# ==============================================================================
st.markdown("""
<div class="main-header">
    <h2 style="color:#ffffff; margin:0; font-size:24px; font-weight:700;">
        FreshHarvest - Fruit Quality Inspection
    </h2>
    <p style="color:#94a3b8; margin:6px 0 0 0; font-size:14px;">
        Automatic fruit freshness detection powered by ResNet50 Transfer Learning.
    </p>
</div>
""", unsafe_allow_html=True)

model, error_msg = load_model_pipeline()

if model is None:
    st.error(f"Error loading model: {error_msg}")
    st.stop()

# ==============================================================================
# 6. Main Inputs
# ==============================================================================
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### Upload Image")
    st.caption("Upload an image file or choose a sample image:")
    
    uploaded_file = st.file_uploader(
        "Upload image (Drag and drop or Browse):",
        type=["jpg", "jpeg", "png", "webp"]
    )
    
    st.markdown("#### Or Select a Sample Image:")
    sample_options = [
        ("None", None),
        ("Fresh Banana", "assets/samples/fresh_banana.jpg"),
        ("Spoiled Banana", "assets/samples/spoiled_banana.jpg"),
        ("Fresh Orange", "assets/samples/fresh_orange.jpg"),
        ("Spoiled Orange", "assets/samples/spoiled_orange.jpg"),
        ("Fresh Strawberry", "assets/samples/fresh_strawberry.jpg"),
        ("Spoiled Strawberry", "assets/samples/spoiled_strawberry.jpg")
    ]
    
    selected_sample_label = st.selectbox(
        "Choose sample:",
        [s[0] for s in sample_options],
        index=0
    )
    
    target_image = None
    source_name = ""
    
    if uploaded_file is not None:
        target_image = Image.open(uploaded_file)
        source_name = uploaded_file.name
    elif selected_sample_label != "None":
        for label, path in sample_options:
            if label == selected_sample_label and path and os.path.exists(path):
                target_image = Image.open(path)
                source_name = f"Sample: {label}"
                break
                
    if target_image is not None:
        st.image(
            target_image, 
            caption=f"Input Image: {source_name} ({target_image.width}x{target_image.height} px)", 
            use_container_width=True
        )
    else:
        st.info("Upload an image or pick a sample to start inspection.")

# ==============================================================================
# 7. Predictions & Results
# ==============================================================================
with col_results:
    st.markdown("### Inspection Results")
    
    if target_image is None:
        st.write("Waiting for image input.")
    else:
        with st.spinner("Running model prediction..."):
            from tensorflow.keras.applications.resnet50 import preprocess_input
            
            t0 = time.time()
            img_resized = target_image.convert("RGB").resize((128, 128))
            img_arr = np.array(img_resized, dtype=np.float32)
            img_batch = np.expand_dims(img_arr, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            
            predictions = model.predict(img_preprocessed, verbose=0)[0]
            latency_ms = (time.time() - t0) * 1000
            
            pred_idx = int(np.argmax(predictions))
            pred_code = CLASS_NAMES[pred_idx]
            confidence = float(predictions[pred_idx] * 100)
            meta = CLASS_DISPLAY[pred_code]
            is_fresh = meta['status'] == 'Fresh'
            
        badge_class = "status-fresh" if is_fresh else "status-spoiled"
        status_label = meta['status'].upper()
        bar_color = "#10b981" if is_fresh else "#ef4444"
        
        st.markdown(f"""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:20px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:12px; color:#94a3b8; text-transform:uppercase; font-weight:600;">Prediction</span>
                <span class="{badge_class}">{status_label}</span>
            </div>
            <div style="margin-top:10px;">
                <div style="font-size:24px; font-weight:700; color:#ffffff;">
                    {meta['status']} {meta['fruit']}
                </div>
                <div style="color:#94a3b8; font-size:13px; font-family:monospace; margin-top:2px;">
                    Class: {pred_code} ({meta['grade']})
                </div>
            </div>
            <div style="margin-top:16px;">
                <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                    <span style="color:#cbd5e1;">Confidence</span>
                    <span style="font-weight:700; color:#ffffff;">{confidence:.2f}%</span>
                </div>
                <div style="background:#334155; height:8px; border-radius:4px; overflow:hidden;">
                    <div style="width:{min(confidence, 100):.1f}%; height:100%; background:{bar_color}; border-radius:4px;"></div>
                </div>
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:10px;">
                Latency: {latency_ms:.1f} ms
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action Notice using native Streamlit alerts
        if is_fresh:
            st.success("Action: Fruit is fresh. Approved for packaging and distribution.")
        else:
            st.error("Action: Fruit is spoiled. Reject to avoid contamination.")
            
        # Top-5 Class Probabilities
        st.markdown("#### Top Class Probabilities")
        top5_indices = np.argsort(predictions)[-5:][::-1]
        top5_data = []
        for idx in top5_indices:
            code = CLASS_NAMES[idx]
            prob = float(predictions[idx] * 100)
            status = CLASS_DISPLAY[code]['status']
            name = f"{status} {CLASS_DISPLAY[code]['fruit']} ({code})"
            top5_data.append({"Class": name, "Probability (%)": prob, "Status": status})
            
        df_top5 = pd.DataFrame(top5_data)
        
        if HAS_PLOTLY:
            fig = px.bar(
                df_top5,
                x="Probability (%)",
                y="Class",
                orientation="h",
                color="Status",
                color_discrete_map={"Fresh": "#10b981", "Spoiled": "#ef4444"},
                text=df_top5["Probability (%)"].apply(lambda v: f"{v:.1f}%")
            )
            fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc", size=12),
                margin=dict(l=0, r=40, t=10, b=10),
                height=190,
                xaxis=dict(showgrid=True, gridcolor="#334155", range=[0, 118], title=""),
                yaxis=dict(autorange="reversed", title="")
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.bar_chart(df_top5.set_index("Class")["Probability (%)"])

        # Quality Details Cards to balance the layout symmetrically
        st.markdown("#### Quality Details")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            q_status_text = "Passed (Grade A)" if is_fresh else "Failed (Defective)"
            q_status_desc = "Meets quality standards" if is_fresh else "Decay detected"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Inspection Status</div>
                <div style="font-size:15px; font-weight:700; color:{bar_color}; margin-top:4px;">
                    {q_status_text}
                </div>
                <div style="font-size:12px; color:#94a3b8; margin-top:2px;">
                    {q_status_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_q2:
            q_action_text = "Cold Storage Room" if is_fresh else "Dispose / Segregate"
            q_action_desc = "Temp: 1°C - 4°C" if is_fresh else "Immediate isolation"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Storage Handling</div>
                <div style="font-size:15px; font-weight:700; color:#f8fafc; margin-top:4px;">
                    {q_action_text}
                </div>
                <div style="font-size:12px; color:#94a3b8; margin-top:2px;">
                    {q_action_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_q3, col_q4 = st.columns(2)
        with col_q3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Detected Fruit</div>
                <div style="font-size:15px; font-weight:700; color:#ffffff; margin-top:4px;">
                    {meta['fruit']}
                </div>
                <div style="font-size:12px; color:#94a3b8; margin-top:2px;">
                    Code: {pred_code}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_q4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Inspection Speed</div>
                <div style="font-size:15px; font-weight:700; color:#ffffff; margin-top:4px;">
                    {latency_ms:.1f} ms
                </div>
                <div style="font-size:12px; color:#94a3b8; margin-top:2px;">
                    ResNet50 Backbone
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("FreshHarvest Quality Inspection - Streamlit and TensorFlow ResNet50")
