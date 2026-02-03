import streamlit as st
import librosa
import numpy as np
import plotly.express as px
import pandas as pd
import tempfile
import os
import parselmouth

# สำหรับลง library 
# pip install streamlit plotly pandas librosa praat-parselmouth numpy
st.set_page_config(page_title="Vocal Health Dashboard", layout="wide")

st.title("🎙️ Vocal Analysis: Pure vs. Combined Visualization")

# --- 1. ส่วนอัปโหลดไฟล์ ---
uploaded_file = st.file_uploader("อัปโหลดไฟล์เสียง (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file:
    # เก็บไฟล์ชั่วคราวเพื่อให้ Librosa อ่าน
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # โหลดข้อมูลเสียง
    y, sr = librosa.load(tmp_file_path)

    # --- 2. สร้าง Tab เพื่อเปรียบเทียบ ---
    tab1, tab2 = st.tabs(["📈 แบบเพียว (Summary Trend)", "🌈 แบบใช้คู่ (Detailed Spectrogram)"])

# --- TAB 1: แบบคำนวณจริง (Real Jitter Calculation) ---
    with tab1:
        st.subheader("📈 Jitter Analysis (Real Calculation)")
        
        # แปลงไฟล์เป็น Object ของ Parselmouth
        snd = parselmouth.Sound(tmp_file_path)
        
        # 1. หาค่า Pitch (ความถี่เสียงร้อง)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        
        # เนื่องจาก Jitter คำนวณเป็นช่วงเวลาไม่ได้ง่ายๆ เหมือน Pitch
        # เราจะแสดง "Pitch Curve" (เส้นเสียงสูงต่ำ) แทน ซึ่งดูง่ายกว่าสำหรับเพลง
        # และคำนวณ Jitter รวม (Overall Jitter) มาโชว์เป็นตัวเลขครับ
        
        st.write("เส้นแสดงระดับเสียงสูง-ต่ำ (Pitch Contour)")
        
        # แก้ปัญหา: กรองค่าที่เป็น 0 (ช่วงเงียบ) ออก เพื่อให้กราฟไม่ตกขอบ
        pitch_values[pitch_values == 0] = np.nan 
        
        # สร้าง Time Axis
        times = pitch.xs()
        
        df_real = pd.DataFrame({"Time (s)": times, "Pitch (Hz)": pitch_values})
        
        # วาดกราฟ Pitch จริงๆ จากเพลง
        fig_real = px.line(df_real, x="Time (s)", y="Pitch (Hz)", 
                           title=f"Melody Contour: {uploaded_file.name}")
        st.plotly_chart(fig_real, use_container_width=True)

        # 2. คำนวณค่า Jitter รวม (ค่าความแหบเฉลี่ยทั้งเพลง)
        st.write("---")
        st.write("### 🩺 ผลตรวจสุขภาพเสียง (เฉลี่ยทั้งไฟล์)")
        
        try:
            # คำนวณ Jitter (local)
            point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
            local_jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # แปลงเป็นเปอร์เซ็นต์
            jitter_pct = local_jitter * 100 
            
            col1, col2 = st.columns(2)
            col1.metric("Jitter (ความแหบ)", f"{jitter_pct:.2f}%")
            
            if jitter_pct > 1.04:
                col2.error("⚠️ สูงกว่าเกณฑ์ (1.04%) - เสียงอาจจะแหบหรือมีดนตรีรบกวนมาก")
            else:
                col2.success("✅ ปกติ")
                
        except:
            st.warning("ไม่สามารถคำนวณ Jitter ได้ (อาจเพราะเสียงดนตรีดังกลบเสียงร้อง)")

    # --- TAB 2: แบบใช้คู่ (Librosa + Plotly) ---
    with tab2:
        st.subheader("แบบที่ 2: Librosa + Plotly (X-Ray เจาะลึก)")
        st.write("ใช้ประมวลผลสัญญาณขั้นสูง (STFT) เพื่อดูโครงสร้างเสียง")
        
        with st.spinner("กำลังทำ STFT และ Mel-filtering..."):
            # ขั้นตอน Librosa (ประมวลผลสัญญาณขั้นสูง)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # ขั้นตอน Plotly (วาด Matrix ที่ Librosa คำนวณมา)
            fig_spec = px.imshow(S_dB, origin='lower', aspect='auto', 
                                 color_continuous_scale='Magma',
                                 labels=dict(x="Time", y="Frequency", color="dB"))
            st.plotly_chart(fig_spec, use_container_width=True)
            st.info("💡 นี่คือ Librosa + Plotly: Librosa แกะรหัสเสียงเป็น 'ตารางสี' แล้ว Plotly ระบายสีออกมา")

    # ลบไฟล์ชั่วคราว
    os.remove(tmp_file_path)
else:
    st.warning("กรุณาอัปโหลดไฟล์เสียงก่อนครับ")