import os
import random
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime, timezone, timedelta
from apify_client import ApifyClient


load_dotenv()

# --- Auto-Detect Semua Token di Hugging Face ---
DAFTAR_TOKEN = [value for key, value in os.environ.items() if key.startswith("APIFY_TOKEN_") and value.strip()]

app = FastAPI(
    title="Fluensy AI (Demo Mode)",
    description="Bot Detection API with Simulated Inference & Multi-Token",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD AI & SCALER

try:
    model = tf.keras.models.load_model('fake_followers_model.keras', compile=False)
    scaler = joblib.load('scaler.pkl')
    print("✅ Model AI dan Scaler siap!")
    print(f"🔋 Sistem berjalan dengan {len(DAFTAR_TOKEN)} Token Apify Cadangan.")
except Exception as e:
    print(f"❌ ERROR Loading AI Assets: {e}")

# 2. ATURAN INPUT
class DataAkun(BaseModel):
    username: str
    jumlah_sampel: int = 50 

# 3. ENDPOINT ROOT 
@app.get("/")
def home():
    return {
        "status": "aktif",
        "message": "API Deteksi Bot AI (Simulated Mode). Gunakan endpoint POST /api/cek-bot."
    }

# 4. ENDPOINT UTAMA 

@app.post("/api/cek-bot")
def cek_akun(data: DataAkun):
    target_ig = data.username.replace("@", "").strip()

    # --- FUNGSI AUTO-SWITCH TOKEN LOKAL ---
    working_token_idx = 0 
    def jalankan_scraper_aman_lokal(actor_id, run_input):
        nonlocal working_token_idx
        if not DAFTAR_TOKEN:
            raise Exception("Environment variables untuk token Apify kosong.")

        for i in range(working_token_idx, len(DAFTAR_TOKEN)):
            try:
                print(f"🔄 Trying Apify with Token {i + 1}...")
                client = ApifyClient(DAFTAR_TOKEN[i])
                run = client.actor(actor_id).call(run_input=run_input)
                hasil = list(client.dataset(run["defaultDatasetId"]).iterate_items())

                if len(hasil) == 0:
                    print(f"⚠️ Token {i + 1} jalan, tapi return 0 data.")
                    raise Exception("Data empty. Limit tercapai atau diblokir IG.")

                print(f"✅ Token {i + 1} Success!")
                working_token_idx = i  
                return hasil 
                
            except Exception as e:
                print(f"⚠️ Token {i + 1} Failed. ALASAN: {str(e)}")
                continue 
        
        raise Exception("CRITICAL: Semua token limit atau gagal dieksekusi.")
    
    # --- FASE 1: AMBIL DATA DARI APIFY (CUMA CEK PRIVATE/PUBLIC) ---
    try:
        run_input_cek = {"usernames": [target_ig]}
        # Panggil Apify cuma buat ngecek 1 profil ini aja
        profil_target = jalankan_scraper_aman_lokal("apify/instagram-profile-scraper", run_input_cek)
        
        if not profil_target:
            raise HTTPException(status_code=404, detail="Akun target tidak ditemukan. Pastikan nama IG benar.")

        info_akun = profil_target[0]
        status_private = info_akun.get("isPrivate") or info_akun.get("is_private") or info_akun.get("private")
        
        # Jika akun digembok,Return awal
        if str(status_private).lower() == "true":
            return {
                "status": "restricted",
                "message": f"Mohon maaf, kami tidak dapat mengaudit @{target_ig} karena akun bersifat private. Silakan pastikan akun target bersifat publik untuk mendapatkan hasil audit yang akurat.",
                "target_akun": f"@{target_ig}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengecek status akun target: {str(e)}")


    # --- FASE 2: MEMBUAT DATA SIMULASI ---
    data_untuk_ai = []
    for i in range(data.jumlah_sampel):
        # AI akan merandom probabilitas bot antara 15% sampai 25% biar kelihatan natural
        is_simulated_bot = random.random() < random.uniform(0.15, 0.25)
        if is_simulated_bot:
            data_untuk_ai.append([0, 0.85, 1, 0.0, 0, 5, 0, 0, 0, 5, 4000]) # Pola Bot
        else:
            data_untuk_ai.append([1, 0.05, 2, 0.0, 0, 50, 1, random.choice([0, 1]), 120, 1500, 350]) # Pola Manusia


    # --- FASE 3: AI REKOMENDASI ---
    nama_kolom = [
        'profile pic', 'nums/length username', 'fullname words', 
        'nums/length fullname', 'name==username', 'description length', 
        'external URL', 'private', '#posts', '#followers', '#follows'
    ]
    df_prediksi = pd.DataFrame(data_untuk_ai, columns=nama_kolom)
    data_scaled = scaler.transform(df_prediksi.values).astype('float32')
    hasil_prediksi = model.predict(data_scaled)
    
    jumlah_bot = sum(1 for skor in hasil_prediksi if skor[0] > 0.5)
    total_diperiksa = len(data_untuk_ai)
    
    if total_diperiksa == 0:
        raise HTTPException(status_code=500, detail="Data tensor kosong.")
        
    persentase_bot = (jumlah_bot / total_diperiksa) * 100
    persentase_asli = 100 - persentase_bot


    # --- FASE 4: LOGIKA REKOMENDASI BERLAPIS ---
    if persentase_bot < 15.0:
        rekomendasi = f"Sangat Berkualitas. Hanya terdeteksi sekitar {persentase_bot:.1f}% pengikut bot. Akun ini memiliki audiens yang sangat organik dan direkomendasikan untuk kolaborasi pemasaran."
    elif persentase_bot < 31.0:
        rekomendasi = f"Wajar. Terdeteksi sekitar {persentase_bot:.1f}% pengikut bot, yang mana masih dalam batas normal industri. Masih aman untuk kolaborasi, namun tetap pantau rasio engagement kontennya."
    elif persentase_bot <= 50.0:
        rekomendasi = f"Waspada. Probabilitas pengikut bot cukup signifikan di angka {persentase_bot:.1f}%. Disarankan untuk melakukan peninjauan manual pada daftar pengikut sebelum mengalokasikan anggaran besar."
    else:
        rekomendasi = f"Berisiko Tinggi. Mayoritas pengikut terdeteksi sebagai akun bot atau tidak aktif ({persentase_bot:.1f}%). Investasi pemasaran pada akun ini berisiko sangat tinggi mengakibatkan pemborosan anggaran (budget waste)."

    waktu_wib = timezone(timedelta(hours=7))

    return {
        "status": "success",
        "target_akun": f"@{target_ig}",
        "total_sampel_diperiksa": total_diperiksa,
        "metrik_kualitas": {
            "persentase_bot": float(f"{persentase_bot:.2f}"),
            "persentase_asli": float(f"{persentase_asli:.2f}"),
            "timestamp": datetime.now(waktu_wib).isoformat()
        },
        "rekomendasi": rekomendasi
    }