import streamlit as st
import re
import numpy as np
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa Ayarları
st.set_page_config(page_title="Türkçe Metin Analiz Dashboard", layout="wide")

# --- Session State Başlatma (Hata Önleyici) ---
if 'nlp_res' not in st.session_state:
    st.session_state.nlp_res = None
if 'read_res' not in st.session_state:
    st.session_state.read_res = None

# --- Fonksiyonlar ---

def check_turkish_spelling(words):
    kalin_unluler, ince_unluler = set("aıou"), set("eiöü")
    return [w for w in words if any(c in kalin_unluler for c in w.lower()) and any(c in ince_unluler for c in w.lower())]

def calculate_readability(text):
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    if not words or not sentences: return None
    
    total_words, total_sentences = len(words), len(sentences)
    total_syllables = sum(len(re.findall(r'[aeıioöuüAEIİOÖUÜ]', w)) for w in words)
    
    avg_w_syl = total_syllables / total_words
    avg_s_word = total_words / total_sentences

    atesman = 199 - (1.015 * avg_s_word) - (42.332 * avg_w_syl)
    cetinkaya = 118.8 - (25.9 * avg_w_syl) - (0.9 * avg_s_word)
    
    return {"at": round(atesman, 2), "cu": round(cetinkaya, 2), "asw": round(avg_s_word, 2), "aws": round(avg_w_syl, 2)}

def get_atesman_info(score):
    if score is None: return "Veri Yok", "❓"
    if score >= 90: return "Çok Kolay", "🍀"
    if score >= 70: return "Kolay", "✅"
    if score >= 50: return "Orta Güçlük", "⚠️"
    if score >= 30: return "Zor", "🚩"
    return "Çok Zor", "🚫"

# --- ARAYÜZ ---
st.title("🇹🇷 Türkçe Metin Analiz Dashboard")

col_in1, col_in2 = st.columns(2)
ai_text = col_in1.text_area("AI Metni (Aday):", value="Yapay zeka modelleri günümüzde çok hızlı gelişiyor.", height=150)
ref_text = col_in2.text_area("Referans Metin (Cevap Anahtarı):", value="Günümüzde yapay zeka sistemleri oldukça süratli bir gelişim gösteriyor.", height=150)

c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 4])

# NLP ANALİZ BUTONU
if c_btn1.button("🚀 NLP Analizi"):
    if ai_text and ref_text:
        with st.spinner("NLP Hesaplanıyor..."):
            P, R, F1 = score([ai_text], [ref_text], lang="tr", verbose=False)
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            cos_sim = cosine_similarity([model.encode(ai_text)], [model.encode(ref_text)])[0][0]
            
            f1_val = F1.mean().item()
            comment = "Anlamsal benzerlik çok yüksek, AI referansı başarıyla yakalamış." if f1_val > 0.85 else "Anlamsal olarak bazı farklılıklar mevcut."
            
            st.session_state.nlp_res = f"--- NLP METRİKLERİ ---\nBERTScore P: {P.mean().item():.4f}\nBERTScore R: {R.mean().item():.4f}\nBERTScore F1: {f1_val:.4f}\nCosine Sim: {cos_sim:.4f}\n\n--- ANALİST YORUMU ---\n{comment}"

# OKUNABİLİRLİK BUTONU
if c_btn2.button("📊 Okunabilirlik Analizi"):
    if ai_text and ref_text:
        r_ai = calculate_readability(ai_text)
        r_ref = calculate_readability(ref_text)
        
        diff = abs(r_ai['at'] - r_ref['at'])
        comment = "AI metni, referans metnin dil ağırlığını tam olarak yansıtıyor." if diff < 10 else "AI metni referansa göre farklı bir karmaşıklık düzeyinde."
        
        st.session_state.read_res = {"ai": r_ai, "ref": r_ref, "comment": comment}

if c_btn3.button("🗑️ Temizle"):
    st.session_state.nlp_res = None
    st.session_state.read_res = None
    st.rerun()

st.divider()

# --- SONUÇLARI GÖSTERME ---
res_left, res_right = st.columns(2)

with res_left:
    st.subheader("🛠️ Teknik NLP Çıktısı")
    if st.session_state.nlp_res:
        st.code(st.session_state.nlp_res, language="text")
    else:
        st.info("Analiz için yukarıdaki butona basın.")

with res_right:
    st.subheader("📈 Okunabilirlik Dashboard")
    if st.session_state.read_res and isinstance(st.session_state.read_res, dict):
        r = st.session_state.read_res
        # Hata korumalı veri çekme
        ai_data = r.get("ai", {})
        ref_data = r.get("ref", {})
        
        m1, m2 = st.columns(2)
        label_ai, icon_ai = get_atesman_info(ai_data.get("at"))
        label_ref, icon_ref = get_atesman_info(ref_data.get("at"))
        
        m1.metric("AI Ateşman", f"{ai_data.get('at')}", f"{label_ai} {icon_ai}", delta_color="off")
        m2.metric("Ref Ateşman", f"{ref_data.get('at')}", f"{label_ref} {icon_ref}", delta_color="off")
        
        st.table({
            "Ölçüt": ["Ateşman İndeksi", "Çetinkaya-Uzun", "Ort. Cümle (Kelime)", "Ort. Kelime (Hece)"],
            "AI Metni": [ai_data.get("at"), ai_data.get("cu"), ai_data.get("asw"), ai_data.get("aws")],
            "Referans": [ref_data.get("at"), ref_data.get("cu"), ref_data.get("asw"), ref_data.get("aws")]
        })
        st.success(f"**💡 Analist Yorumu:** {r.get('comment')}")
    else:
        st.info("Okunabilirlik analizi bekleniyor...")
