import os
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

# Setup fallback tokens dari env
DAFTAR_TOKEN = [
    os.getenv("APIFY_TOKEN_1"),
    os.getenv("APIFY_TOKEN_2"),
    os.getenv("APIFY_TOKEN_3"),
    os.getenv("APIFY_TOKEN_4"),
    os.getenv("APIFY_TOKEN_5"),
    os.getenv("APIFY_TOKEN_6"),
    os.getenv("APIFY_TOKEN_7")
]
# Filter token kosong
DAFTAR_TOKEN = [token for token in DAFTAR_TOKEN if token]

app = FastAPI(
    title="AI",
    description="Bot Detection Inference Engine",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML assets on startup
try:
    model = tf.keras.models.load_model('fake_followers_model.keras', compile=False)
    scaler = joblib.load('scaler.pkl')
    print("✅ Model & Scaler loaded.")
    print(f"🔋 Active Apify Tokens: {len(DAFTAR_TOKEN)}")
except Exception as e:
    print(f"❌ Failed to load ML assets: {e}")

class DataAkun(BaseModel):
    username: str

@app.get("/")
def home():
    return {"status": "up", "service": " AI Engine"}

@app.post("/api/cek-bot")
def cek_akun(data: DataAkun):
    target_ig = data.username.replace("@", "").strip()
    BATAS_SAMPEL = 50 

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
                    print(f"⚠️ Token {i + 1} jalan, tapi return 0 data (Kemungkinan Limit).")
                    raise Exception("Data empty. Limit tercapai.")

                print(f"✅ Token {i + 1} Success!")
                working_token_idx = i  
                return hasil 
                
            except Exception as e:
                print(f"⚠️ Token {i + 1} Failed. Pindah ke token berikutnya...")
                continue 
        
        raise Exception("CRITICAL: Semua token limit atau kena block.")

    # 1. Cek Private Account
    try:
        run_input_cek = {"usernames": [target_ig]}
        profil_target = jalankan_scraper_aman_lokal("apify/instagram-profile-scraper", run_input_cek)
        
        if not profil_target:
            raise HTTPException(status_code=404, detail="Akun tidak ditemukan.")

        info_akun = profil_target[0]
        status_private = info_akun.get("isPrivate") or info_akun.get("is_private") or info_akun.get("private")
        
        if str(status_private).lower() == "true":
            return {
                "status": "restricted",
                "message": f"Tidak bisa memproses @{target_ig}. Akun di-private.",
                "target_akun": f"@{target_ig}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Target check failed: {str(e)}")

    # 2. Get Followers List
    try:
        run_input_followers = {
            "Account": [target_ig], 
            "getFollowers": True,
            "getFollowing": False,
            "resultsLimit": BATAS_SAMPEL 
        }
        followers_mentah = jalankan_scraper_aman_lokal("scraping_solutions/instagram-scraper-followers-following-no-cookies", run_input_followers)
        daftar_username = [f.get("username") for f in followers_mentah if f.get("username")]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraper 1 failed: {str(e)}")

    if not daftar_username:
        raise HTTPException(status_code=404, detail="Gagal ekstrak followers list.")

    # 3. Get Profiles Detail
    try:
        run_input_profiles = {"usernames": daftar_username}
        profil_lengkap = jalankan_scraper_aman_lokal("apify/instagram-profile-scraper", run_input_profiles)

        if not profil_lengkap:
            raise HTTPException(status_code=500, detail="Profile details empty.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraper 2 failed: {str(e)}")

    # 4. Feature Extraction untuk NN Model
    data_untuk_ai = []
    for prof in profil_lengkap:
        uname = prof.get("username", "").strip()
        fname = prof.get("fullName", "").strip() or uname
        bio = prof.get("biography", "")
        is_private = prof.get("isPrivate", False)
        pic_url = prof.get("profilePicUrl", "")

        punya_foto = 1 if pic_url and "default" not in pic_url else 0

        len_uname = len(uname)
        nums_length_username = sum(c.isdigit() for c in uname) / len_uname if len_uname > 0 else 0.0 
        len_fname = len(fname)
        nums_length_fullname = sum(c.isdigit() for c in fname) / len_fname if len_fname > 0 else 0.0 
        fullname_words = len(fname.split()) 
        name_equals_username = 1 if uname.lower() == fname.lower() else 0 
        
        description_length = len(bio)
        external_url = 1 if prof.get("externalUrl") else 0
        private_status = 1 if is_private else 0 
        
        posts_count = int(prof.get("postsCount", 0))
        followers_count = int(prof.get("followersCount", 0))
        follows_count = int(prof.get("followsCount", 0))

        data_untuk_ai.append([
            punya_foto, nums_length_username, fullname_words, nums_length_fullname, 
            name_equals_username, description_length, external_url, private_status, 
            posts_count, followers_count, follows_count
        ])

    # 5. Predict using tf.keras model
    nama_kolom = [
        'profile pic', 'nums/length username', 'fullname words', 
        'nums/length fullname', 'name==username', 'description length', 
        'external URL', 'private', '#posts', '#followers', '#follows'
    ]
    
    df_prediksi = pd.DataFrame(data_untuk_ai, columns=nama_kolom)
    data_scaled = scaler.transform(df_prediksi.values).astype('float32')
    hasil_prediksi = model.predict(data_scaled)
    
    # Hitung rasio (Threshold 0.5)
    jumlah_bot = sum(1 for skor in hasil_prediksi if skor[0] > 0.5)
    total_diperiksa = len(data_untuk_ai)
    
    if total_diperiksa == 0:
        raise HTTPException(status_code=500, detail="Data tensor kosong.")
        
    persentase_bot = (jumlah_bot / total_diperiksa) * 100
    persentase_asli = 100 - persentase_bot

    # 6. Response Builder
    if persentase_bot < 15.0:
        rekomendasi = f"Sangat Berkualitas. Hanya terdeteksi sekitar {persentase_bot:.1f}% pengikut bot. Akun ini memiliki audiens yang sangat organik dan direkomendasikan untuk kolaborasi pemasaran."
    elif persentase_bot < 31.0:
        rekomendasi = f"Wajar. Terdeteksi sekitar {persentase_bot:.1f}% pengikut bot, yang mana masih dalam batas normal industri. Masih aman untuk kolaborasi, namun tetap pantau rasio engagement kontennya."
    elif persentase_bot <= 50.0:
        rekomendasi = f"Waspada. Probabilitas pengikut bot cukup signifikan di angka {persentase_bot:.1f}%. Disarankan untuk melakukan peninjauan manual pada daftar pengikut sebelum mengalokasikan anggaran besar."
    else:
        rekomendasi = f"Berisiko Tinggi. Mayoritas pengikut terdeteksi sebagai akun bot atau tidak aktif ({persentase_bot:.1f}%). Investasi pemasaran pada akun ini berisiko sangat tinggi mengakibatkan pemborosan anggaran (budget waste)."

    return {
        "status": "success",
        "target_akun": f"@{target_ig}",
        "total_sampel_diperiksa": total_diperiksa,
        "metrik_kualitas": {
            "persentase_bot": float(f"{persentase_bot:.2f}"),
            "persentase_asli": float(f"{persentase_asli:.2f}"),
            "timestamp": datetime.now(timezone(timedelta(hours=7))).isoformat()
        },
        "rekomendasi": rekomendasi
    }