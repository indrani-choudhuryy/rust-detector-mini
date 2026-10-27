import cv2
import numpy as np
import streamlit as st

from rust_detector import analyze_rust

st.set_page_config(page_title="Rust Detector Mini", page_icon="🔍", layout="wide")
st.title("🔍 Rust Detector Mini")
st.caption("Upload an image of a pipe or metal surface to estimate its corrosion level.")

with st.sidebar:
    st.header("Settings")
    min_sat = st.slider("Min saturation (lower = more sensitive)", 20, 200, 80)
    min_val = st.slider("Min brightness", 0, 150, 40)
    st.info("If the background is brownish, try increasing the saturation threshold.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the image. Please try a different file.")
    else:
        percent, level, mask, overlay = analyze_rust(image, min_sat, min_val)

        colors = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
        c1, c2 = st.columns(2)
        c1.metric("Rust coverage", f"{percent:.2f}%")
        c2.metric("Corrosion level", f"{colors[level]} {level}")

        col_a, col_b, col_c = st.columns(3)
        col_a.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        col_b.image(mask, caption="Rust mask", use_container_width=True)
        col_c.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Rust highlighted", use_container_width=True)
else:
    st.write("👆 Upload an image to get started.")
