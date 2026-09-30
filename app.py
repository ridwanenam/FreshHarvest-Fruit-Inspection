import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. Page Configuration & Theme
# ==============================================================================
st.set_page_config(
    page_title="FreshHarvest - AI Fruit Quality Inspection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.4) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    .status-badge-fresh {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 9999px;
        display: inline-block;
        font-size: 14px;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    }
    
    .status-badge-spoiled {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
        color: #ffffff;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 9999px;
        display: inline-block;
        font-size: 14px;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
    }
    
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    
    .kpi-number {
        font-size: 26px;
        font-weight: 800;
        color: #38bdf8;
    }
    
    .kpi-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    
    .dispatch-box-fresh {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 5px solid #10b981;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 16px;
    }
    
    .dispatch-box-spoiled {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 5px solid #ef4444;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. Target Classes & Metadata
# ==============================================================================
CLASS_NAMES = [
    'F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry', 'F_Tamarillo', 'F_Tomato',
    'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango', 'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato'
]

CLASS_DISPLAY = {
    'F_Banana': {'fruit': 'Banana', 'status': 'Fresh', 'icon': '🍌', 'grade': 'Grade A - Premium'},
    'F_Lemon': {'fruit': 'Lemon', 'status': 'Fresh', 'icon': '🍋', 'grade': 'Grade A - Premium'},
    'F_Lulo': {'fruit': 'Lulo (Naranjilla)', 'status': 'Fresh', 'icon': '🍈', 'grade': 'Grade A - Premium'},
    'F_Mango': {'fruit': 'Mango', 'status': 'Fresh', 'icon': '🥭', 'grade': 'Grade A - Premium'},
    'F_Orange': {'fruit': 'Orange', 'status': 'Fresh', 'icon': '🍊', 'grade': 'Grade A - Premium'},
    'F_Strawberry': {'fruit': 'Strawberry', 'status': 'Fresh', 'icon': '🍓', 'grade': 'Grade A - Premium'},
    'F_Tamarillo': {'fruit': 'Tamarillo', 'status': 'Fresh', 'icon': '🍅', 'grade': 'Grade A - Premium'},
    'F_Tomato': {'fruit': 'Tomato', 'status': 'Fresh', 'icon': '🍅', 'grade': 'Grade A - Premium'},
    'S_Banana': {'fruit': 'Banana', 'status': 'Spoiled', 'icon': '🍌', 'grade': 'Reject - Bio-hazard'},
    'S_Lemon': {'fruit': 'Lemon', 'status': 'Spoiled', 'icon': '🍋', 'grade': 'Reject - Bio-hazard'},
    'S_Lulo': {'fruit': 'Lulo (Naranjilla)', 'status': 'Spoiled', 'icon': '🍈', 'grade': 'Reject - Bio-hazard'},
    'S_Mango': {'fruit': 'Mango', 'status': 'Spoiled', 'icon': '🥭', 'grade': 'Reject - Bio-hazard'},
    'S_Orange': {'fruit': 'Orange', 'status': 'Spoiled', 'icon': '🍊', 'grade': 'Reject - Bio-hazard'},
    'S_Strawberry': {'fruit': 'Strawberry', 'status': 'Spoiled', 'icon': '🍓', 'grade': 'Reject - Bio-hazard'},
    'S_Tamarillo': {'fruit': 'Tamarillo', 'status': 'Spoiled', 'icon': '🍅', 'grade': 'Reject - Bio-hazard'},
    'S_Tomato': {'fruit': 'Tomato', 'status': 'Spoiled', 'icon': '🍅', 'grade': 'Reject - Bio-hazard'}
}

# ==============================================================================
# 3. Model Loader (Cached & Resilient)
# ==============================================================================
@st.cache_resource(show_spinner="Loading ResNet50 Transfer Learning Model...")
def load_model_pipeline():
    candidate_paths = [
        "models/resnet50_freshharvest_transfer_model.keras",
        "../week_6/resnet50_freshharvest_transfer_model.keras",
        "resnet50_freshharvest_transfer_model.keras",
        "../models/resnet50_freshharvest_transfer_model.keras"
    ]
    model_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if not model_path:
        return None, f"Model file not found. Checked paths: {candidate_paths}"
        
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        # Load with custom_objects and safe_mode=False for Keras 3 compatibility
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"preprocess_input": preprocess_input},
            safe_mode=False
        )
        return model, None
    except Exception as e:
        return None, str(e)

# ==============================================================================
# 4. Sidebar: Hardware & Production Telemetry
# ==============================================================================
with st.sidebar:
    st.markdown("### 🏢 FreshHarvest Logistics")
    st.markdown("**Automated Cold Storage Inspection**  \n*California Facility #4*")
    st.markdown("---")
    
    st.markdown("#### ⚡ AI Telemetry")
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
            <div class="kpi-label">Sorting KPI</div>
            <div class="kpi-number" style="color:#38bdf8;">99.05%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Backbone Model</div>
        <div style="font-size:15px; font-weight:700; color:#f8fafc; margin-top:4px;">ResNet50 (Transfer Learning)</div>
        <div style="font-size:12px; color:#94a3b8; margin-top:2px;">Frozen Weights (ImageNet) + Custom Head</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 📦 Supported Produce (16 Classes)")
    st.caption("8 Fruits in Fresh (F_) & Spoiled (S_) condition:")
    produce_list = ["Banana", "Lemon", "Lulo", "Mango", "Orange", "Strawberry", "Tamarillo", "Tomato"]
    st.write(", ".join(produce_list))
    
    st.markdown("---")
    st.caption("FreshHarvest AI Deployment • Week 6 Streamlit Demo")

# ==============================================================================
# 5. Header Section
# ==============================================================================
st.markdown("""
<div class="main-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1 style="color:#ffffff; margin:0; font-size:28px; font-weight:800; letter-spacing:-0.5px;">
                🍎 FreshHarvest AI — Fruit Freshness Inspection
            </h1>
            <p style="color:#cbd5e1; margin:6px 0 0 0; font-size:15px;">
                Real-time automated sorting demo powered by <b>ResNet50 Transfer Learning</b> for cold storage conveyor facilities.
            </p>
        </div>
        <div>
            <span class="status-badge-fresh">SYSTEM READY</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load the model
model, error_msg = load_model_pipeline()

if model is None:
    st.error(f"⚠️ Model Loading Error: {error_msg}")
    st.info("Make sure 'resnet50_freshharvest_transfer_model.keras' is located in the 'models/' folder.")
    st.stop()

# ==============================================================================
# 6. Main Interface: Drag & Drop + Sample Tester
# ==============================================================================
col_input, col_results = st.columns([1.1, 1.3], gap="large")

with col_input:
    st.markdown("### 📤 Upload Produce Image")
    st.caption("Drag and drop an image file or test with warehouse pre-loaded samples:")
    
    # Drag-and-drop file uploader
    uploaded_file = st.file_uploader(
        "Upload Fruit Image (Drag and drop or Browse):",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload clear photos of fruits from conveyor cameras"
    )
    
    st.markdown("#### 🧪 Or Try Sample Warehouse Images:")
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
        "Choose a sample image:",
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
            caption=f"Source: {source_name} ({target_image.width}x{target_image.height} px)", 
            use_container_width=True
        )
    else:
        st.markdown("""
        <div style="border: 2px dashed #475569; border-radius: 12px; padding: 48px 24px; text-align: center; background: rgba(30, 41, 59, 0.4); margin-top: 16px;">
            <div style="font-size: 40px; margin-bottom: 8px;">📷</div>
            <div style="color: #94a3b8; font-size: 15px; font-weight: 500;">Please drag & drop an image or select a sample above.</div>
            <div style="color: #64748b; font-size: 12px; margin-top: 4px;">Supported formats: JPG, PNG, WEBP</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 7. Real-Time Inference & Analytics Pipeline
# ==============================================================================
with col_results:
    st.markdown("### 🔍 Real-Time Inspection Results")
    
    if target_image is None:
        st.info("Awaiting image input to execute inspection analysis.")
    else:
        with st.spinner("Analyzing produce texture and freshness signatures..."):
            from tensorflow.keras.applications.resnet50 import preprocess_input
            
            t0 = time.time()
            # 1. Resize & Preprocess for ResNet50
            img_resized = target_image.convert("RGB").resize((128, 128))
            img_arr = np.array(img_resized, dtype=np.float32)
            img_batch = np.expand_dims(img_arr, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            
            # 2. Run Inference
            predictions = model.predict(img_preprocessed, verbose=0)[0]
            latency_ms = (time.time() - t0) * 1000
            
            # 3. Compute Metrics
            pred_idx = int(np.argmax(predictions))
            pred_code = CLASS_NAMES[pred_idx]
            confidence = float(predictions[pred_idx] * 100)
            meta = CLASS_DISPLAY[pred_code]
            is_fresh = meta['status'] == 'Fresh'
            
        # Display Decision Card
        if is_fresh:
            badge_html = f'<span class="status-badge-fresh">✅ {meta["status"].upper()}</span>'
            status_color = "#10b981"
            dispatch_html = f"""
            <div class="dispatch-box-fresh">
                <div style="font-weight:700; color:#10b981; font-size:16px;">
                    ✅ DISPATCH DECISION: ROUTE TO COLD CONVEYOR A
                </div>
                <div style="color:#cbd5e1; font-size:13px; margin-top:4px;">
                    Produce meets <b>Grade A supermarket quality requirements</b>. Approved for immediate packaging and distribution.
                </div>
            </div>
            """
        else:
            badge_html = f'<span class="status-badge-spoiled">⚠️ {meta["status"].upper()} (DEFECT DETECTED)</span>'
            status_color = "#ef4444"
            dispatch_html = f"""
            <div class="dispatch-box-spoiled">
                <div style="font-weight:700; color:#ef4444; font-size:16px;">
                    ⚠️ DISPATCH DECISION: REJECT TO DIVERTER B
                </div>
                <div style="color:#cbd5e1; font-size:13px; margin-top:4px;">
                    Fruit exhibits <b>decay/spoilage characteristics</b>. Automatic pneumatic ejector activated to protect batch freshness.
                </div>
            </div>
            """
            
        st.markdown(f"""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:14px; padding:24px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-size:14px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">
                    Detected Classification
                </div>
                {badge_html}
            </div>
            <div style="display:flex; align-items:center; gap:12px; margin-top:12px;">
                <span style="font-size:38px;">{meta['icon']}</span>
                <div>
                    <div style="font-size:26px; font-weight:800; color:#ffffff;">
                        {meta['status']} {meta['fruit']}
                    </div>
                    <div style="color:#64748b; font-size:13px; font-family:monospace;">
                        Internal Code: {pred_code} • {meta['grade']}
                    </div>
                </div>
            </div>
            <div style="margin-top:20px;">
                <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:6px;">
                    <span style="color:#cbd5e1; font-weight:600;">Model Confidence Score</span>
                    <span style="color:{status_color}; font-weight:800; font-size:15px;">{confidence:.2f}%</span>
                </div>
                <div style="background:#334155; height:10px; border-radius:9999px; overflow:hidden;">
                    <div style="width:{min(confidence, 100):.1f}%; height:100%; background:{status_color}; border-radius:9999px;"></div>
                </div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px; color:#94a3b8; margin-top:14px;">
                <span>⚡ Latency: <b>{latency_ms:.1f} ms</b></span>
                <span>🎯 Baseline KPI: <b>98.68%</b></span>
            </div>
            {dispatch_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Top-5 Class Probability Distribution
        st.markdown("#### 📊 Top Candidate Probabilities")
        top5_indices = np.argsort(predictions)[-5:][::-1]
        top5_data = []
        for idx in top5_indices:
            code = CLASS_NAMES[idx]
            prob = float(predictions[idx] * 100)
            status = CLASS_DISPLAY[code]['status']
            name = f"{status} {CLASS_DISPLAY[code]['fruit']} ({code})"
            top5_data.append({"Class": name, "Probability (%)": prob, "Status": status})
            
        df_top5 = pd.DataFrame(top5_data)
        
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", family="Inter"),
            margin=dict(l=0, r=20, t=10, b=10),
            height=240,
            xaxis=dict(showgrid=True, gridcolor="#334155", range=[0, 100]),
            yaxis=dict(autorange="reversed", title="")
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("FreshHarvest Cold Storage Quality Assurance System • Built with Streamlit & TensorFlow ResNet50")
