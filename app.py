import streamlit as st
import pandas as pd
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
from pytesseract import Output

# --------------------------------------------------
# Config
# --------------------------------------------------
st.set_page_config(page_title="HYROX OCR → CSV", layout="centered")
st.title("HYROX OCR Tool (Internal)")

st.write(
    "Carica **2 screenshot Roxfit**:\n"
    "- uno con **RUNS**\n"
    "- uno con **STATIONS**\n\n"
    "Il tool esegue OCR robusto e genera un CSV."
)

# --------------------------------------------------
# Utility
# --------------------------------------------------
def mmss_to_sec(t):
    m, s = t.split(":")
    return int(m) * 60 + int(s)

# --------------------------------------------------
# OCR RUNS – flessibile e robusto
# --------------------------------------------------
from pytesseract import Output

def extract_runs_flexible(pil_img):
    img = np.array(pil_img)
    h, w, _ = img.shape

    # crop status bar (molto conservativo)
    img = img[int(h * 0.08):h, :]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    data = pytesseract.image_to_data(
        thresh,
        output_type=Output.DATAFRAME,
        config="--psm 6 -c tessedit_char_whitelist=0123456789:"
    )

    data = data.dropna(subset=["text"])
    data["text"] = data["text"].str.strip()

    # mm:ss
    mask = data["text"].str.match(r"^\d{2}:\d{2}$")
    data = data[mask]

    if data.empty:
        return []

    # in secondi
    data["sec"] = data["text"].apply(mmss_to_sec)

    # filtro fisiologico run HYROX
    data = data[(data["sec"] > 150) & (data["sec"] < 420)]

    if data.empty:
        return []

    # coordinate
    data["x_center"] = data["left"] + data["width"] / 2

    # --- TENTATIVO CLUSTER COLONNE ---
    try:
        data["col"] = pd.qcut(data["x_center"], q=3, duplicates="drop")
        counts = data["col"].value_counts()

        if counts.empty:
            raise ValueError

        best_col = counts.idxmax()
        runs = data[data["col"] == best_col]

    except Exception:
        # FALLBACK: una sola colonna → ordina tutto verticalmente
        runs = data

    # ordine verticale
    runs = runs.sort_values("top")

    return runs["text"].tolist()


# --------------------------------------------------
# OCR STATIONS – semplice (ordine HYROX)
# --------------------------------------------------
def extract_station_times(pil_img):
    img = np.array(pil_img)
    h, w, _ = img.shape

    img = img[int(h * 0.08):h, :]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 6 -c tessedit_char_whitelist=0123456789:"
    )

    times = re.findall(r"\d{2}:\d{2}", text)

    # filtro fisiologico stazioni
    out = []
    for t in times:
        sec = mmss_to_sec(t)
        if 30 < sec < 600:
            out.append(t)

    return out

# --------------------------------------------------
# Upload
# --------------------------------------------------
runs_img = st.file_uploader("Screenshot RUNS", type=["png", "jpg", "jpeg"])
stations_img = st.file_uploader("Screenshot STATIONS", type=["png", "jpg", "jpeg"])

if runs_img and stations_img and st.button("Processa e genera CSV"):

    with st.spinner("OCR in corso..."):
        run_times = extract_runs_flexible(Image.open(runs_img))
        station_times = extract_station_times(Image.open(stations_img))

    results = []

    # -------- RUNS --------
    if len(run_times) != 8:
        st.warning(f"Trovate {len(run_times)} runs (attese 8)")

    for i, t in enumerate(run_times[:8]):
        results.append((f"Run_{i+1}", mmss_to_sec(t)))

    # -------- STATIONS --------
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

    for name, t in zip(station_order, station_times):
        results.append((name, mmss_to_sec(t)))

    if not results:
        st.error("Nessun dato riconosciuto.")
    else:
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

