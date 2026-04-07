import streamlit as st
import re
import numpy as np
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa Ayarları
st.set_page_config(page_title="Türkçe Metin Analiz Dashboard", layout="wide")

# --- Session State Başlatma ---
if 'nlp_res' not in st.session_state: st.session_state.nlp_res = None
if 'read_res' not in st.session_state: st.session_state.read_res = None

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

def get_cetinkaya_info(score):
    if score is None: return "Veri Yok", "❓"
    if score >= 80: return "Eğitim düzeyi düşük", "📗"
    if score >= 60: return "Orta eğitim düzeyi", "📘"
    if score >= 40: return "Yüksek eğitim düzeyi", "📙"
    return "Akademik/Uzman düzey", "📕"

# --- ARAYÜZ ---
st.title("🇹🇷 Gelişmiş Türkçe Metin Analiz Dashboard")

col_in1, col_in2 = st.columns(2)
ai_text = col_in1.text_area("AI Metni (Aday):", value="Yapay zeka modelleri günümüzde çok hızlı gelişiyor.", height=150)
ref_text = col_in2.text_area("Referans Metin (Cevap Anahtarı):", value="Günümüzde yapay zeka sistemleri oldukça süratli bir gelişim gösteriyor.", height=150)

c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 4])

# --- NLP ANALİZ MANTIĞI ---
if c_btn1.button("🚀 NLP Analizi"):
    if ai_text and ref_text:
        with st.spinner("NLP Hesaplanıyor..."):
            ai_words = re.findall(r'\b\w+\b', ai_text, re.UNICODE)
            ref_words = re.findall(r'\b\w+\b', ref_text, re.UNICODE)
            ai_sentences = [s for s in re.split(r'[.!?]+', ai_text) if s.strip()]
            ref_sentences = [s for s in re.split(r'[.!?]+', ref_text) if s.strip()]
            ai_spell_issues = check_turkish_spelling(ai_words)
            P, R, F1 = score([ai_text], [ref_text], lang="tr", verbose=False)
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            cos_sim = cosine_similarity([model.encode(ai_text)], [model.encode(ref_text)])[0][0]
            f1_val = F1.mean().item()
            
            raw_report = f"--- TEMEL ÖZELLİKLER ---\n[AI] Kelime: {len(ai_words)} | Cümle: {len(ai_sentences)}\n"
            raw_report += f"[Ref] Kelime: {len(ref_words)} | Cümle: {len(ref_sentences)}\n\n"
            raw_report += "--- İMLA KONTROLÜ (AI) ---\n"
            raw_report += (", ".join(ai_spell_issues) + "\n\n") if ai_spell_issues else "Sorun yok.\n\n"
            raw_report += f"--- TEKNİK METRİKLER ---\nPrecision : {P.mean().item():.4f}\nRecall    : {R.mean().item():.4f}\nF1-Score  : {f1_val:.4f}\nCosine    : {cos_sim:.4f}"
            
            comment = "Anlamsal uyum mükemmel." if f1_val > 0.90 else "Anlamsal farklılıklar var."
            st.session_state.nlp_res = {"raw": raw_report, "f1": f1_val, "cos": cos_sim, "comment": comment}

# --- OKUNABİLİRLİK MANTIĞI ---
if c_btn2.button("📊 Okunabilirlik Analizi"):
    if ai_text and ref_text:
        r_ai, r_ref = calculate_readability(ai_text), calculate_readability(ref_text)
        st.session_state.read_res = {"ai": r_ai, "ref": r_ref}

if c_btn3.button("🗑️ Temizle"):
    st.session_state.nlp_res = st.session_state.read_res = None
    st.rerun()

st.divider()

# --- GÖRSELLEŞTİRME ---
res_left, res_right = st.columns(2)

with res_left:
    st.subheader("🛠️ NLP Analiz Raporu")
    if st.session_state.nlp_res and isinstance(st.session_state.nlp_res, dict):
        n = st.session_state.nlp_res
        st.code(n.get('raw'), language="text")
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("F1 (Başarı)", f"{n.get('f1'):.2%}")
        m_c2.metric("Anlam Benzerliği", f"{n.get('cos'):.2%}")
        st.success(f"**💡 NLP Yorumu:** {n.get('comment')}")
    else:
        st.info("NLP analizi bekleniyor...")

with res_right:
    st.subheader("📈 Okunabilirlik Dashboard")
    if st.session_state.read_res and isinstance(st.session_state.read_res, dict):
        r = st.session_state.read_res
        ai_d, ref_d = r.get("ai", {}), r.get("ref", {})

        # Ateşman Bölümü
        st.markdown("**Ateşman İndeksi (Genel)**")
        m1, m2 = st.columns(2)
        l_ai, i_ai = get_atesman_info(ai_d.get("at"))
        l_ref, i_ref = get_atesman_info(ref_d.get("at"))
        m1.metric("AI Ateşman", f"{ai_d.get('at')}", f"{l_ai} {i_ai}", delta_color="off")
        m2.metric("Ref Ateşman", f"{ref_d.get('at')}", f"{l_ref} {i_ref}", delta_color="off")

        # Çetinkaya-Uzun Bölümü
        st.markdown("**Çetinkaya-Uzun (Eğitim Seviyesi)**")
        m3, m4 = st.columns(2)
        cl_ai, ci_ai = get_cetinkaya_info(ai_d.get("cu"))
        cl_ref, ci_ref = get_cetinkaya_info(ref_d.get("cu"))
        m3.metric("AI Çetinkaya", f"{ai_d.get('cu')}", f"{cl_ai} {ci_ai}", delta_color="off")
        m4.metric("Ref Çetinkaya", f"{ref_d.get('at')}", f"{cl_ref} {ci_ref}", delta_color="off")

        st.table({
            "Ölçüt": ["Ateşman", "Çetinkaya-Uzun", "Cümle/Kelime", "Hece/Kelime"],
            "AI": [ai_d.get("at"), ai_d.get("cu"), ai_d.get("asw"), ai_d.get("aws")],
            "Ref": [ref_d.get("at"), ref_d.get("cu"), ref_d.get("asw"), ref_d.get("aws")]
        })
    else:
        st.info("Okunabilirlik analizi bekleniyor...")
