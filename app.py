from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Online Shopper Segmentation",
    page_icon="🛒",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
]

LOG_FEATURES = FEATURES[:6]

CLUSTER_INFO = {
    0: {
        "name": "Product-Focused Explorers",
        "description": "Sesi yang cukup aktif menjelajahi halaman produk dengan exit rate rendah.",
        "recommendation": "Berikan rekomendasi produk dan perjelas alur menuju checkout.",
    },
    1: {
        "name": "Immediate Bouncers",
        "description": "Sesi dengan aktivitas sangat rendah serta bounce rate dan exit rate tinggi.",
        "recommendation": "Periksa relevansi landing page, kecepatan website, dan kesesuaian sumber traffic.",
    },
    2: {
        "name": "Deeply Engaged Researchers",
        "description": "Sesi dengan engagement paling tinggi pada halaman informasi dan produk.",
        "recommendation": "Sediakan informasi lengkap, perbandingan produk, ulasan, dan call to action yang jelas.",
    },
    3: {
        "name": "Brief Returning Browsers",
        "description": "Sesi penjelajahan relatif singkat yang banyak berasal dari returning visitor.",
        "recommendation": "Tampilkan kembali produk yang pernah dilihat dan mudahkan pengguna melanjutkan aktivitas sebelumnya.",
    },
}


@st.cache_resource
def load_model():
    scaler = joblib.load(BASE_DIR / "online_shoppers_scaler.joblib")
    kmeans = joblib.load(BASE_DIR / "online_shoppers_kmeans.joblib")
    return scaler, kmeans


scaler, kmeans = load_model()

st.title("Segmentasi Perilaku Sesi Pengunjung E-Commerce")
st.write(
    "Masukkan ringkasan aktivitas satu sesi untuk mengetahui kelompok perilakunya. "
    "Hasil aplikasi merupakan segmentasi, bukan prediksi terjadinya transaksi."
)

with st.expander("Tentang model"):
    st.write(
        "Model K-Means membagi 12.330 sesi menjadi empat segmen berdasarkan "
        "delapan fitur aktivitas pengunjung. Model memperoleh Silhouette Score "
        "0,4240 dan Davies-Bouldin Index 0,8162."
    )
    st.write(
        "Revenue tidak digunakan untuk membentuk cluster. Hasil aplikasi menunjukkan "
        "segmen terdekat dan rekomendasinya perlu diuji sebelum diterapkan."
    )
    st.markdown(
        "Sumber data: [UCI Online Shoppers Purchasing Intention Dataset]"
        "(https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)"
    )

with st.form("session_form"):
    admin_col, info_col, product_col = st.columns(3)

    with admin_col:
        administrative = st.number_input(
            "Jumlah halaman administratif", min_value=0, max_value=27, value=1
        )
        administrative_duration = st.number_input(
            "Durasi halaman administratif (detik)",
            min_value=0.0,
            max_value=3398.75,
            value=7.5,
        )

    with info_col:
        informational = st.number_input(
            "Jumlah halaman informasional", min_value=0, max_value=24, value=0
        )
        informational_duration = st.number_input(
            "Durasi halaman informasional (detik)",
            min_value=0.0,
            max_value=2549.38,
            value=0.0,
        )

    with product_col:
        product_related = st.number_input(
            "Jumlah halaman produk", min_value=0, max_value=705, value=18
        )
        product_related_duration = st.number_input(
            "Durasi halaman produk (detik)",
            min_value=0.0,
            max_value=63973.52,
            value=598.94,
        )

    bounce_col, exit_col = st.columns(2)

    with bounce_col:
        bounce_rates = st.number_input(
            "Bounce rate",
            min_value=0.0,
            max_value=0.2,
            value=0.003,
            step=0.001,
            format="%.3f",
            help="Gunakan nilai desimal, misalnya 0,05 untuk 5%.",
        )

    with exit_col:
        exit_rates = st.number_input(
            "Exit rate",
            min_value=0.0,
            max_value=0.2,
            value=0.025,
            step=0.001,
            format="%.3f",
            help="Gunakan nilai desimal, misalnya 0,05 untuk 5%.",
        )

    submitted = st.form_submit_button("Identifikasi Segmen", type="primary")

if submitted:
    page_duration_inputs = [
        ("administratif", administrative, administrative_duration),
        ("informasional", informational, informational_duration),
        ("produk", product_related, product_related_duration),
    ]
    invalid_inputs = [
        page_type
        for page_type, page_count, duration in page_duration_inputs
        if page_count == 0 and duration > 0
    ]

    if invalid_inputs:
        st.warning(
            "Durasi halaman harus 0 ketika jumlah halaman bernilai 0. "
            f"Periksa kembali input halaman {', '.join(invalid_inputs)}."
        )
    else:
        input_data = pd.DataFrame(
            [[
                administrative,
                administrative_duration,
                informational,
                informational_duration,
                product_related,
                product_related_duration,
                bounce_rates,
                exit_rates,
            ]],
            columns=FEATURES,
        )

        input_data[LOG_FEATURES] = np.log1p(input_data[LOG_FEATURES])
        input_scaled = scaler.transform(input_data)
        cluster = int(kmeans.predict(input_scaled)[0])
        result = CLUSTER_INFO[cluster]

        st.success(f"Segmen terdekat: {result['name']}")
        st.write(result["description"])
        st.write(f"Rekomendasi: {result['recommendation']}")
