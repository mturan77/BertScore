import streamlit as st
import re
import numpy as np
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa Ayarları
st.set_page_config(page_title="Türkçe NLP & Okunabilirlik Dashboard", layout="wide")

# --- Session State Başlatma (Hata Önleyici) ---
if 'nlp_res' not in st.session_state:
    st.session_state.nlp_res = None
if 'read_res' not in st.session_state:
    st.session_state.read_res = None

# --- Fonksiyonlar ---

def check_turkish_spelling(words):
    kalin, ince = set("aıou"), set("eiöü")
    return [w for w in words if any(c in kalin for c in w.lower()) and any(c in ince for c in w.lower())]

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
st.title("🇹🇷 Gelişmiş Türkçe Metin Analiz Aracı")

col_in1, col_in2 = st.columns(2)
ai_text = col_in1.text_area("AI Metni (Aday):", value="Yapay zeka modelleri günümüzde çok hızlı gelişiyor.", height=150)
ref_text = col_in2.text_area("Referans Metin (Cevap Anahtarı):", value="Günümüzde yapay zeka sistemleri oldukça süratli bir gelişim gösteriyor.", height=150)

c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 4])

# --- NLP ANALİZ MANTIĞI ---
if c_btn1.button("🚀 NLP Analizi"):
    if ai_text and ref_text:
        with st.spinner("Modeller yükleniyor ve hesaplanıyor..."):
            ai_words = re.findall(r'\b\w+\b', ai_text, re.UNICODE)
            ref_words = re.findall(r'\b\w+\b', ref_text, re.UNICODE)
            ai_sentences = [s for s in re.split(r'[.!?]+', ai_text) if s.strip()]
            ref_sentences = [s for s in re.split(r'[.!?]+', ref_text) if s.strip()]
            ai_spell_issues = check_turkish_spelling(ai_words)
            
            P, R, F1 = score([ai_text], [ref_text], lang="tr", verbose=False)
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            cos_sim = cosine_similarity([model.encode(ai_text)], [model.encode(ref_text)])[0][0]
            
            f1_val = F1.mean().item()
            
            # Teknik Ham Rapor
            raw_report = "--- TEMEL ÖZELLİKLER ---\n"
            raw_report += f"[AI Cümlesi]       Kelime: {len(ai_words)} | Cümle: {len(ai_sentences)}\n"
            raw_report += f"[Referans Cümle] Kelime: {len(ref_words)} | Cümle: {len(ref_sentences)}\n\n"
            raw_report += "--- İMLA KONTROLÜ (AI) ---\n"
            raw_report += (", ".join(ai_spell_issues) + "\n\n") if ai_spell_issues else "Yazım sorunu tespit edilmedi.\n\n"
            raw_report += "--- TEKNİK METRİKLER ---\n"
            raw_report += f"BERTScore Precision : {P.mean().item():.4f}\n"
            raw_report += f"BERTScore Recall    : {R.mean().item():.4f}\n"
            raw_report += f"BERTScore F1-Score  : {f1_val:.4f}\n"
            raw_report += f"Cosine Similarity   : {cos_sim:.4f}\n"

            # Yorum
            if f1_val > 0.90: comment = "Mükemmel! Metinler anlamsal olarak özdeş."
            elif f1_val > 0.75: comment = "Başarılı. AI, referansın ana fikrini korumuş."
            else: comment = "Zayıf. AI metni anlam kaybı yaşıyor veya çok farklı ifade edilmiş."

            st.session_state.nlp_res = {"raw": raw_report, "f1": f1_val, "cos": cos_sim, "comment": comment}

# --- OKUNABİLİRLİK MANTIĞI ---
if c_btn2.button("📊 Okunabilirlik Analizi"):
    if ai_text and ref_text:
        r_ai, r_ref = calculate_readability(ai_text), calculate_readability(ref_text)
        diff = abs(r_ai['at'] - r_ref['at'])
        comment = "AI metni, referansın dil ağırlığına tam uyum sağlıyor." if diff < 10 else "AI metni referanstan farklı bir seviyede."
        st.session_state.read_res = {"ai": r_ai, "ref": r_ref, "comment": comment}

if c_btn3.button("🗑️ Temizle"):
    st.session_state.nlp_res = None
    st.session_state.read_res = None
    st.rerun()

st.divider()

# --- GÖRSELLEŞTİRME ---
res_left, res_right = st.columns(2)

with res_left:
    st.subheader("🛠️ NLP Analiz Raporu")
    # Hata korumalı kontrol: nlp_res hem var olmalı hem de sözlük olmalı
    if st.session_state.nlp_res and isinstance(st.session_state.nlp_res, dict):
        n = st.session_state.nlp_res
        st.code(n.get('raw', 'Veri alınamadı'), language="text")
        
        st.write("**NLP Başarı Özeti:**")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Genel Başarı (F1)", f"{n.get('f1', 0):.2%}")
        m_col2.metric("Anlam Benzerliği", f"{n.get('cos', 0):.2%}")
        st.success(f"**💡 NLP Yorumu:** {n.get('comment', '')}")
    else:
        st.info("NLP analizi bekleniyor...")

with res_right:
    st.subheader("📈 Okunabilirlik Dashboard")
    if st.session_state.read_res and isinstance(st.session_state.read_res, dict):
        r = st.session_state.read_res
        ai_data = r.get("ai", {})
        ref_data = r.get("ref", {})
        
        m1, m2 = st.columns(2)
        l_ai, i_ai = get_atesman_info(ai_data.get("at"))
        l_ref, i_ref = get_atesman_info(ref_data.get("at"))
        
        m1.metric("AI Ateşman", f"{ai_data.get('at')}", f"{l_ai} {i_ai}", delta_color="off")
        m2.metric("Ref Ateşman", f"{ref_data.get('at')}", f"{l_ref} {i_ref}", delta_color="off")
        
        st.table({
            "Ölçüt": ["Ateşman Puanı", "Çetinkaya-Uzun", "Ort. Cümle", "Ort. Hece"],
            "AI Metni": [ai_data.get("at"), ai_data.get("cu"), ai_data.get("asw"), ai_data.get("aws")],
            "Referans": [ref_data.get("at"), ref_data.get("cu"), ref_data.get("asw"), ref_data.get("aws")]
        })
        st.info(f"**💡 Okunabilirlik Yorumu:** {r.get('comment', '')}")
    else:
        st.info("Okunabilirlik analizi bekleniyor...")
