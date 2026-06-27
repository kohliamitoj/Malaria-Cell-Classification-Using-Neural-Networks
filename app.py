import os
import numpy as np
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MalariaScope | AI Cell Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 3rem 3rem; max-width: 1200px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0D1B3E 0%, #0A2342 60%, #06141E 100%);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 18px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(6,182,212,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 700;
    background: linear-gradient(90deg, #38BDF8, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
    line-height: 1.2;
}
.hero .subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    margin: 0 0 1.2rem;
}
.badge {
    display: inline-block;
    background: rgba(6,182,212,0.12);
    color: #38BDF8;
    border: 1px solid rgba(6,182,212,0.28);
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.2rem;
}

/* ── Cards ── */
.card {
    background: #0F172A;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.6rem;
}

/* ── Section titles ── */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #CBD5E1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}

/* ── Result banners ── */
.result-infected {
    background: linear-gradient(135deg, rgba(127,29,29,0.6), rgba(153,27,27,0.4));
    border: 1px solid rgba(239,68,68,0.6);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.result-healthy {
    background: linear-gradient(135deg, rgba(6,78,59,0.6), rgba(6,95,70,0.4));
    border: 1px solid rgba(16,185,129,0.6);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.result-sub {
    font-size: 0.85rem;
    margin-top: 0.25rem;
    opacity: 0.75;
}

/* ── Metric tiles ── */
.metric-tile {
    background: #0F172A;
    border: 1px solid rgba(6,182,212,0.18);
    border-radius: 12px;
    padding: 1.2rem 0.8rem;
    text-align: center;
}
.metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #38BDF8;
    line-height: 1;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.4rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] section {
    background: #0F172A !important;
    border: 2px dashed rgba(56,189,248,0.25) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(56,189,248,0.55) !important;
}

/* ── Placeholder box ── */
.placeholder {
    height: 280px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #334155;
    border: 1px dashed #1E293B;
    border-radius: 12px;
    gap: 0.5rem;
}
.placeholder-icon { font-size: 2.5rem; }
.placeholder-text { font-size: 0.9rem; }

/* ── Warning note ── */
.disclaimer {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid rgba(245,158,11,0.5);
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #94A3B8;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading VGG19 model…")
def load_model():
    model_path = "vgg19_malaria.keras"

    if not os.path.exists(model_path):
        # Download from Google Drive using the file ID stored in Streamlit secrets
        try:
            import gdown
            file_id = st.secrets.get("GDRIVE_MODEL_ID", "")
            if not file_id:
                return None, "Model file not found locally and GDRIVE_MODEL_ID secret is not set."
            gdown.download(id=file_id, output=model_path, quiet=False)
        except Exception as exc:
            return None, str(exc)

    try:
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as exc:
        return None, str(exc)


def predict(model, pil_img: Image.Image):
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.vgg19.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    raw = float(model.predict(arr, verbose=0)[0][0])

    # flow_from_dataframe assigns class indices alphabetically:
    # 'parasitized' → 0,  'uninfected' → 1
    # sigmoid output ≈ P(uninfected)
    if raw >= 0.5:
        label, confidence = "Uninfected", raw
    else:
        label, confidence = "Parasitized", 1.0 - raw

    return label, round(confidence * 100, 1)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔬 MalariaScope</h1>
    <p class="subtitle">AI-powered malaria detection from blood cell microscopy images</p>
    <div>
        <span class="badge">VGG19 Transfer Learning</span>
        <span class="badge">TensorFlow 2.16</span>
        <span class="badge">27,558 Cell Images</span>
        <span class="badge">~94% Val Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)

model, err = load_model()

if err and not model:
    st.error(f"Could not load model: {err}")
    st.info("Run the notebook first to generate `vgg19_malaria.keras`, then restart this app.")
    st.stop()

# ── Classifier section ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="section-title">📤 Upload Cell Image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, use_container_width=True, caption="Uploaded microscopy image")

with right:
    st.markdown('<p class="section-title">🧬 Prediction</p>', unsafe_allow_html=True)

    if uploaded and model:
        with st.spinner("Analysing cell…"):
            label, confidence = predict(model, pil_img)

        is_infected = label == "Parasitized"
        css_class = "result-infected" if is_infected else "result-healthy"
        icon = "⚠️" if is_infected else "✅"
        color_hex = "#EF4444" if is_infected else "#10B981"
        sub_text = "Malaria parasite detected" if is_infected else "No parasite detected"

        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-label" style="color:{color_hex}">{icon} {label}</div>
            <div class="result-sub" style="color:{color_hex}">{sub_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 40, "color": "#F1F5F9", "family": "Inter"}},
            title={"text": "Confidence", "font": {"size": 14, "color": "#64748B"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#475569"}},
                "bar": {"color": color_hex, "thickness": 0.28},
                "bgcolor": "#1E293B",
                "bordercolor": "#334155",
                "borderwidth": 1,
                "steps": [{"range": [0, 100], "color": "#0F172A"}],
                "threshold": {
                    "line": {"color": color_hex, "width": 2},
                    "thickness": 0.8,
                    "value": confidence,
                },
            },
        ))
        fig.update_layout(
            height=230,
            margin=dict(l=30, r=30, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="disclaimer">
            ⚠️ This tool is for <strong>educational and demonstration purposes only</strong>.
            It is not a certified medical diagnostic device.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">🩸</div>
            <div class="placeholder-text">Upload an image on the left to see the result</div>
        </div>
        """, unsafe_allow_html=True)

# ── Key metrics ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-title">📊 Best Model Performance — VGG19</p>', unsafe_allow_html=True)

cols = st.columns(4, gap="medium")
tiles = [
    ("93.92%", "Val Accuracy"),
    ("93.39%", "Precision"),
    ("94.58%", "Recall"),
    ("93.98%", "F1 Score"),
]
for col, (val, lbl) in zip(cols, tiles):
    with col:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Model comparison chart ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-title">📈 Model Comparison</p>', unsafe_allow_html=True)

model_names  = ["Custom CNN", "ResNet50", "VGG19 ★", "InceptionV3"]
val_accuracy = [61.19, 92.02, 93.92, 49.84]
val_precision= [60.98, 91.34, 93.39, 69.32]
val_recall   = [65.49, 93.10, 94.58,  2.18]

fig2 = go.Figure()
bar_cfg = dict(marker_line_width=0, width=0.22, opacity=0.9)

fig2.add_trace(go.Bar(name="Val Accuracy",  x=model_names, y=val_accuracy,
                      marker_color="#38BDF8", **bar_cfg))
fig2.add_trace(go.Bar(name="Val Precision", x=model_names, y=val_precision,
                      marker_color="#818CF8", **bar_cfg))
fig2.add_trace(go.Bar(name="Val Recall",    x=model_names, y=val_recall,
                      marker_color="#34D399", **bar_cfg))

fig2.update_layout(
    barmode="group",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94A3B8"),
    yaxis=dict(
        range=[0, 100],
        gridcolor="rgba(255,255,255,0.05)",
        title="Score (%)",
        ticksuffix="%",
    ),
    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    height=360,
    margin=dict(l=0, r=0, t=40, b=0),
    bargap=0.28,
)
st.plotly_chart(fig2, use_container_width=True)

# ── About ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️  About this project"):
    st.markdown("""
    **Malaria Cell Classification** is a deep learning project that applies transfer learning to detect
    malaria parasites in blood cell microscopy images.

    Four CNN architectures were trained and compared on the NIH Malaria Cell Image Dataset — a balanced
    corpus of 27,558 microscopy images (13,779 Parasitized + 13,779 Uninfected):

    | Model | Val Accuracy | Val Precision | Val Recall |
    |---|---|---|---|
    | Custom CNN | 61.19% | 60.98% | 65.49% |
    | ResNet50 | 92.02% | 91.34% | 93.10% |
    | **VGG19** ★ | **93.92%** | **93.39%** | **94.58%** |
    | InceptionV3 | 49.84% | 69.32% | 2.18% |

    **VGG19** with ImageNet pretrained weights (frozen) and a custom Dense head was selected as the
    production model due to its best overall balance of precision and recall.

    **Stack:** TensorFlow 2.16 · Keras 3 · Python 3.11 · Streamlit

    **Dataset:** [NIH Malaria Cell Images via Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
    """)
