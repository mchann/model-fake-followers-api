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

# --- Auto-Detect Semua Token di Hugging Face ---
DAFTAR_TOKEN = [value for key, value in os.environ.items() if key.startswith("APIFY_TOKEN_") and value.strip()]

app = FastAPI(
    title="Fluensy AI",
    description="Bot Detection API with Multi-Token Scraper System",
    version="2.3.0" # Versi Real Data
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# LOAD AI & SCALER
# ==========================================
try:
    model = tf.keras.models.load_model('fake_followers_model.keras', compile=False)
    scaler = joblib.load('scaler.pkl')
    print("✅ Model AI dan Scaler siap!")
    print(f"🔋 Sistem berjalan dengan {len(DAFTAR_TOKEN)} Token Apify Cadangan.")
except Exception as e:
    print(f"❌ ERROR Loading AI Assets: {e}")

# ==========================================
# ATURAN INPUT
# ==========================================
class DataAkun(BaseModel):
    username: str

@app.get("/")
def home():
    return {
        "status": "aktif",
        "message": "Fluensy AI Engine is Running. Real Scraper Mode Active."
    }

# ==========================================
# ENDPOINT UTAMA 
# ==========================================
@app.post("/api/cek-bot")
def cek_akun(data: DataAkun):
    target_ig = data.username.replace("@", "").strip()
    BATAS_SAMPEL = 50 

    # --- FUNGSI AUTO-SWITCH TOKEN LOKAL ---
    working_token_idx = 0 
    def jalankan_scraper_aman_lokal(actor_id, run_input):
        nonlocal working_token_idx
        if not DAFTAR_TOKEN:
            raise Exception("Environment variables untuk token Apify kosong.")

        for i in range(working_token_idx, len(DAFTAR_TOKEN)):
            try:
                print(f"🔄 Trying Apify with Token {i + 1} for {actor_id}...")
                client = ApifyClient(DAFTAR_TOKEN[i])
                run = client.actor(actor_id).call(run_input=run_input)
                
                # Resolusi Bug Subscriptable ('Run' object)
                if isinstance(run, dict):
                    dataset_id = run.get("defaultDatasetId")
                else:
                    dataset_id = getattr(run, "defaultDatasetId", getattr(run, "default_dataset_id", None))
                
                if not dataset_id:
                    raise Exception("Gagal mendapatkan dataset ID dari Apify.")

                hasil = list(client.dataset(dataset_id).iterate_items())

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
    
    # --- FASE 0: CEK PRIVATE/PUBLIC ---
    try:
        run_input_cek = {"usernames": [target_ig]}
        profil_target = jalankan_scraper_aman_lokal("apify/instagram-profile-scraper", run_input_cek)
        
        if not profil_target:
            raise HTTPException(status_code=404, detail="Akun target tidak ditemukan. Pastikan nama IG benar.")

        info_akun = profil_target[0]
        status_private = info_akun.get("isPrivate") or info_akun.get("is_private") or info_akun.get("private")
        
        if str(status_private).lower() == "true":
            return {
                "status": "restricted",
                "message": f"Mohon maaf, kami tidak dapat mengaudit @{target_ig} karena akun bersifat private.",
                "target_akun": f"@{target_ig}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengecek status akun target: {str(e)}")


    # --- FASE 1: SCRAPER PERTAMA (AMBIL 50 FOLLOWERS) ---
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
        raise HTTPException(status_code=500, detail=f"Scraper Followers Failed: {str(e)}")

    if not daftar_username:
        raise HTTPException(status_code=404, detail="Gagal ekstrak daftar followers. Mungkin terblokir.")

    # --- FASE 2: SCRAPER KEDUA (AMBIL DETAIL 50 PROFIL) ---
    try:
        run_input_profiles = {"usernames": daftar_username}
        profil_lengkap = jalankan_scraper_aman_lokal("apify/instagram-profile-scraper", run_input_profiles)

        if not profil_lengkap:
            raise HTTPException(status_code=500, detail="Profile details empty.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraper Profil Failed: {str(e)}")


    # --- FASE 3: EKSTRAKSI FITUR UNTUK AI ---
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

    # --- FASE 4: AI PREDICTION ---
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

    # --- FASE 5: DECISION LOGIC ---
    if persentase_bot < 15.0:
        rekomendasi = f"Sangat Berkualitas. Hanya terdeteksi {persentase_bot:.1f}% pengikut mencurigakan. Audiens sangat organik dan direkomendasikan."
    elif persentase_bot < 31.0:
        rekomendasi = f"Wajar. Terdeteksi {persentase_bot:.1f}% pengikut dengan pola tidak aktif/bot. Masih dalam batas toleransi normal industri."
    elif persentase_bot <= 50.0:
        rekomendasi = f"Waspada. Terdapat indikasi {persentase_bot:.1f}% pengikut memiliki pola tidak wajar. Disarankan melihat metrik engagement (Likes/Comments) secara langsung."
    else:
        rekomendasi = f"Perlu Tinjauan Lanjut. Mayoritas dari sampel ({persentase_bot:.1f}%) terdeteksi tidak aktif. Disarankan untuk meninjau insight audiens influencer secara manual sebelum mengalokasikan anggaran pemasaran."
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