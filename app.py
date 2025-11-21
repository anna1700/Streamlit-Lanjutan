import streamlit as st
import joblib
import pandas as pd

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Spam Slayer AI", page_icon="🛡️")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = joblib.load('spam_detector_model.pkl')
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan 'spam_detector_model.pkl' ada di folder yang sama.")
    st.stop()

# --- SIDEBAR (INFO PROJECT) ---
st.sidebar.title("ℹ️ Project Info")
st.sidebar.info("Aplikasi ini menggunakan **Ensemble Learning** (Voting Classifier).")
st.sidebar.markdown("---")
st.sidebar.write("**Dynamic Duo Algorithms:**")
st.sidebar.write("1. Multinomial Naive Bayes")
st.sidebar.write("2. Random Forest")
st.sidebar.markdown("---")
st.sidebar.metric(label="Akurasi Model", value="96.5%", delta="Super Accuracy")

# --- UTAMA ---
st.title("⚔️ SMS Spam Detector")
st.write("Masukkan pesan SMS atau Email di bawah ini untuk mengecek apakah itu **SPAM** (Penipuan/Iklan) atau **HAM** (Pesan Normal).")

# Input User
user_input = st.text_area("Masukkan Teks Pesan:", height=150, placeholder="Contoh: Congratulations! You've won a $1000 gift card...")

if st.button("🔍 Analisis Pesan"):
    if user_input:
        # Prediksi
        prediction_prob = model.predict_proba([user_input])
        prediction = model.predict([user_input])[0]
        
        # Ambil probabilitas tertinggi
        confidence = prediction_prob[0][prediction] * 100
        
        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 HASIL: TERDETEKSI SPAM! (Yakin {confidence:.2f}%)")
            st.warning("Hati-hati! Pesan ini mengandung indikasi penipuan atau promosi sampah.")
        else:
            st.success(f"✅ HASIL: PESAN AMAN (HAM) (Yakin {confidence:.2f}%)")
            st.info("Pesan ini terlihat seperti percakapan normal.")
    else:
        st.warning("Mohon masukkan teks terlebih dahulu.")
