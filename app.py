import streamlit as st
import os
import io
import json
import time
import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw
from openai import OpenAI
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# CLOUD DETECTION
# --------------------------------------------------
IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "false") == "true"

# --------------------------------------------------
# OPTIONAL HEAVY IMPORTS (LOCAL ONLY)
# --------------------------------------------------
YOLO_AVAILABLE = False
RTDETR_AVAILABLE = False
CLIP_AVAILABLE = False

if not IS_CLOUD:
    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
    except:
        pass

    try:
        from ultralytics import RTDETR
        RTDETR_AVAILABLE = True
    except:
        pass

    try:
        import clip
        import torch
        CLIP_AVAILABLE = True
    except:
        pass

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="🛒 SmartShelf AI",
    layout="wide"
)

# --------------------------------------------------
# API KEY (STREAMLIT CLOUD SAFE)
# --------------------------------------------------
api_key = None

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

if not api_key:
    st.warning("Please provide OpenAI API Key")
    st.stop()

client = OpenAI(api_key=api_key)

# --------------------------------------------------
# ENSEMBLE ANALYZER (CV ONLY – CLOUD SAFE)
# --------------------------------------------------
class EnsembleAnalyzer:
    def extract_color_features(self, image):
        img = np.array(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        masks = {
            "green": cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)),
            "yellow": cv2.inRange(hsv, (20, 80, 80), (35, 255, 255)),
            "brown": cv2.inRange(hsv, (8, 50, 50), (25, 255, 200))
        }

        total = img.shape[0] * img.shape[1]
        pct = {k: np.sum(v > 0) / total for k, v in masks.items()}

        if pct["green"] > 0.4:
            return "unripe", pct["green"]
        if pct["brown"] > 0.2:
            return "overripe", pct["brown"]
        if pct["yellow"] > 0.3:
            return "ripe", pct["yellow"]

        return "unknown", 0.5

# --------------------------------------------------
# WATERSHED BANANA DETECTOR (FAST + ACCURATE)
# --------------------------------------------------
class ShelfAnalyzer:
    def __init__(self):
        self.ensemble = EnsembleAnalyzer()

    def detect(self, image):
        img = np.array(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(hsv, (15, 40, 40), (45, 255, 255))
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
        fg = np.uint8(fg)
        bg = cv2.dilate(mask, kernel, 3)
        unknown = cv2.subtract(bg, fg)

        _, markers = cv2.connectedComponents(fg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), markers)

        detections = []
        idx = 1
        for m in np.unique(markers):
            if m <= 1:
                continue
            cnts, _ = cv2.findContours(
                np.uint8(markers == m),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cnts:
                if cv2.contourArea(c) > 200:
                    x,y,w,h = cv2.boundingRect(c)
                    crop = image.crop((x,y,x+w,y+h))
                    detections.append({
                        "id": idx,
                        "bbox": (x,y,x+w,y+h),
                        "image": crop
                    })
                    idx += 1
        return detections

    def analyze(self, detections):
        bananas = []
        for d in detections:
            ripeness, conf = self.ensemble.extract_color_features(d["image"])
            d["ripeness"] = ripeness
            d["confidence"] = conf
            bananas.append(d)
        return bananas

    def price(self, banana):
        base = {"unripe":35, "ripe":55, "overripe":20}.get(banana["ripeness"], 40)
        discount = 0.3 if banana["ripeness"] == "overripe" else 0
        final = base * (1-discount)
        return round(final,2)

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🛒 SmartShelf AI – Cloud Edition")
st.caption("Fast CV + Watershed | Streamlit Cloud Ready")

if IS_CLOUD:
    st.info("☁️ Running on Streamlit Cloud (Fast Mode Enabled)")
else:
    st.success("🖥️ Local Mode (Pro Models Available)")

analyzer = ShelfAnalyzer()

uploaded = st.file_uploader("Upload shelf image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("🚀 Analyze Shelf"):
        with st.spinner("Analyzing bananas..."):
            dets = analyzer.detect(image)
            bananas = analyzer.analyze(dets)

        st.success(f"🍌 Detected {len(bananas)} banana clusters")

        draw = ImageDraw.Draw(image)
        for b in bananas:
            draw.rectangle(b["bbox"], outline="yellow", width=3)
            draw.text(
                (b["bbox"][0], b["bbox"][1]-15),
                f'{b["ripeness"].upper()}',
                fill="yellow"
            )

        st.image(image, caption="Detection Result", use_container_width=True)

        st.markdown("### 📊 Pricing Summary")
        for b in bananas:
            price = analyzer.price(b)
            st.write(
                f"Cluster #{b['id']} → "
                f"{b['ripeness']} | "
                f"₹{price}/dozen"
            )

        export = {
            "timestamp": datetime.now().isoformat(),
            "total_clusters": len(bananas),
            "bananas": bananas
        }

        st.download_button(
            "📥 Download Report",
            json.dumps(export, indent=2),
            "shelf_report.json",
            "application/json"
        )
