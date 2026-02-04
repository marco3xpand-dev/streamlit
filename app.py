import streamlit as st
import pandas as pd
import pytesseract
import shutil

if shutil.which("tesseract") is None:
    raise RuntimeError("Tesseract non trovato nel sistema")
import cv2
import numpy as np
from PIL import Image
import re

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
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def run_ocr(pil_img):
    img = np.array(pil_img)
    img = preprocess_image(img)
    text = pytesseract.image_to_string(img, config="--psm 6")
    return text.lower()

def mmss_to_sec(t):
    m, s = t.split(":")
    return int(m) * 60 + int(s)

# -----------------------------
# Upload
# -----------------------------
runs_img = st.file_uploader("Screenshot RUNS", type=["png", "jpg", "jpeg"])
stations_img = st.file_uploader("Screenshot STATIONS", type=["png", "jpg", "jpeg"])

if runs_img and stations_img and st.button("Processa e genera CSV"):

    with st.spinner("OCR in corso..."):
        runs_text = run_ocr(Image.open(runs_img))
        stations_text = run_ocr(Image.open(stations_img))

    st.subheader("OCR RAW (debug)")
    st.text_area("Runs OCR", runs_text, height=150)
    st.text_area("Stations OCR", stations_text, height=200)

    results = []

    # -----------------------------
    # Parse RUNS
    # -----------------------------
    run_pattern = re.compile(r"run\s*(\d)\D+(\d{1,2}:\d{2})")

    for m in run_pattern.finditer(runs_text):
        run_id = f"Run_{m.group(1)}"
        time_sec = mmss_to_sec(m.group(2))
        results.append((run_id, time_sec))

    # -----------------------------
    # Parse STATIONS
    # -----------------------------
    station_map = {
        "ski": "SkiErg",
        "sled push": "SledPush",
        "sled pull": "SledPull",
        "bbj": "BurpeeBroadJump",
        "burpee": "BurpeeBroadJump",
        "row": "Rower",
        "farmers": "FarmersCarry",
        "sandbag": "SandbagLunge",
        "wall": "WallBall"
    }

    time_pattern = re.compile(r"\d{1,2}:\d{2}")

    for line in stations_text.split("\n"):
        t = time_pattern.search(line)
        if not t:
            continue

        for key, name in station_map.items():
            if key in line:
                results.append((name, mmss_to_sec(t.group())))
                break

    if len(results) == 0:
        st.error("Nessun dato riconosciuto. OCR troppo sporco.")
    else:
        df = pd.DataFrame(results, columns=["segment", "time_sec"])
        df = df.drop_duplicates("segment", keep="first")

        hyrox_order = [
            "Run_1", "SkiErg",
            "Run_2", "SledPush",
            "Run_3", "SledPull",
            "Run_4", "BurpeeBroadJump",
            "Run_5", "Rower",
            "Run_6", "FarmersCarry",
            "Run_7", "SandbagLunge",
            "Run_8", "WallBall"
        ]

        df["order"] = df["segment"].apply(
            lambda x: hyrox_order.index(x) if x in hyrox_order else 999
        )
        df = df.sort_values("order").drop(columns="order")

        st.subheader("Dati estratti")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Scarica CSV",
            csv,
            file_name="hyrox_race.csv",
            mime="text/csv"
        )
