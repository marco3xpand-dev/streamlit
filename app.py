import streamlit as st
import pandas as pd
import pytesseract
import shutil
import cv2
import numpy as np
from PIL import Image
import re

# -----------------------------
# Check Tesseract
# -----------------------------
if shutil.which("tesseract") is None:
    raise RuntimeError("Tesseract non trovato nel sistema")

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="HYROX OCR → CSV", layout="centered")
st.title("HYROX OCR Tool (Internal)")

st.write(
    "Carica **2 screenshot Roxfit**:\n"
    "- uno con **Runs**\n"
    "- uno con **Stations**\n\n"
    "Il tool esegue OCR e genera un CSV pronto per Excel."
)

# -----------------------------
# OCR helpers
# -----------------------------
def run_ocr(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    text = pytesseract.image_to_string(
        thresh,
        config="--psm 6 -c tessedit_char_whitelist=0123456789:"
    )
    return text.lower()

def mmss_to_sec(t):
    m, s = t.split(":")
    return int(m) * 60 + int(s)

# -----------------------------
# Upload
# -----------------------------
runs_img = st.file_uploader("Screenshot RUNS", type=["png", "jpg", "jpeg"])
stations_img = st.file_uploader("Screenshot STATIONS", type=["png", "jpg", "jpeg"])

# -----------------------------
# Processing
# -----------------------------
if runs_img and stations_img and st.button("Processa e genera CSV"):

    with st.spinner("OCR in corso..."):
        runs_text = run_ocr(Image.open(runs_img))
        stations_text = run_ocr(Image.open(stations_img))

    st.subheader("OCR RAW (debug)")
    st.text_area("Runs OCR", runs_text, height=150)
    st.text_area("Stations OCR", stations_text, height=200)

    TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}")

    def extract_times(text):
        return TIME_PATTERN.findall(text)

    results = []

    # -------- RUNS (per ordine) --------
    run_times = extract_times(runs_text)

    for i, t in enumerate(run_times[:8]):
        results.append((f"Run_{i+1}", mmss_to_sec(t)))

    # -------- STATIONS (per ordine HYROX) --------
    station_order = [
        "SkiErg",
        "SledPush",
        "SledPull",
        "BurpeeBroadJump",
        "Rower",
        "FarmersCarry",
        "SandbagLunge",
        "WallBall"
    ]

    station_times = extract_times(stations_text)

    for name, t in zip(station_order, station_times):
        results.append((name, mmss_to_sec(t)))

    if len(results) < 16:
        st.warning("Attenzione: numero di segmenti inferiore al previsto.")

    df = pd.DataFrame(results, columns=["segment", "time_sec"])

    st.subheader("Dati estratti")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Scarica CSV",
        csv,
        file_name="hyrox_race.csv",
        mime="text/csv"
    )

