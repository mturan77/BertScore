import streamlit as st
import re
import numpy as np
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa Ayarları
st.set_page_config(page_title="Türkçe NLP & Okunabilirlik Analizi", layout="wide")

# --- Hafıza (Session State) Tanımlama ---
if 'nlp_res' not in st.session_state:
    st.session_state.nlp_res = None
if 'read_res' not in st.session_state:
    st.session_state.read_res = None

# --- Fonksiyonlar ---

def check_turkish_spelling(words):
    kalin_unluler = set("aıou")
    ince_unluler = set("eiöü")
    issues = []
    for word in words:
        lower_word = word.lower()
        has_kalin = any(c in kalin_unluler for c in lower_word)
        has_ince = any(c in ince_unluler for c in lower_word)
        if has_kalin and has_ince:
            issues.append(word)
    return issues

def calculate_readability_metrics(text):
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    if not words or not sentences: return None
    
    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(len(re.findall(r'[aeıioöuüAEIİOÖUÜ]', w)) for w in words)
    
    avg_word_len_syllable = total_syllables / total_words
    avg_sentence_len_word = total_words / total_sentences

    atesman = 199 - (1.015 * avg_sentence_len_word) - (42.332 * avg_word_len_syllable)
    cetinkaya = 118.8 - (25.9 * avg_word_len_syllable) - (0.9 * avg_sentence_len_word)
    
    return {
        "atesman": round(atesman, 2), "cetinkaya": round(cetinkaya, 2),
        "avg_sentence": round(avg_sentence_len_word, 2), "avg_syllable": round(avg_word_len_syllable, 2)
    }

def get_atesman_desc(score):
    if score >= 90: return "Çok Kolay (90-100)"
    if score >= 70: return "Kolay (70-89)"
    if score >= 50: return "Orta Güçlük (50-69)"
    if score >= 30: return "Zor (30-49)"
    return "Çok Zor (0-29)"

# --- ARAYÜZ ---
st.title("Türkçe Metin Analiz Aracı")

ai_text = st.text_area("AI Cümlesi (Aday / Üretilen Metin):", value="Yapay zeka modelleri günümüzde çok hızlı gelişiyor.", height=120)
ref_text = st.text_area("Cevap Anahtarı (Referans Metin):", value="Günümüzde yapay zeka sistemleri oldukça süratli bir gelişim gösteriyor.", height=120)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
analyze_nlp_btn = col_btn1.button("NLP Analizi Yap")
analyze_read_btn = col_btn2.button("Okunabilirlik Analizi Yap")
if col_btn3.button("Tüm Sonuçları Temizle"):
    st.session_state.nlp_res = None
    st.session_state.read_res = None
    st.rerun()

# --- Hesaplama Mantığı ---

if analyze_nlp_btn:
    if ai_text and ref_text:
        with st.spinner("NLP Analizi yapılıyor..."):
            ai_words = re.findall(r'\b\w+\b', ai_text, re.UNICODE)
            ref_words = re.findall(r'\b\w+\b', ref_text, re.UNICODE)
            ai_sentences = [s for s in re.split(r'[.!?]+', ai_text) if s.strip()]
            ref_sentences = [s for s in re.split(r'[.!?]+', ref_text) if s.strip()]
            ai_spell_issues = check_turkish_spelling(ai_words)

            P, R, F1 = score([ai_text], [ref_text], lang="tr", verbose=False)
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            embeddings = model.encode([ai_text, ref_text])
            cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            res = "--- TEMEL ÖZELLİKLER ---\n"
            res += f"[AI Cümlesi]       Kelime Sayısı: {len(ai_words)} | Cümle Sayısı: {len(ai_sentences)}\n"
            res += f"[Referans Cümle] Kelime Sayısı: {len(ref_words)} | Cümle Sayısı: {len(ref_sentences)}\n\n"
            res += "--- İMLA & YAZIM KONTROLÜ (AI Cümlesi) ---\n"
            res += (", ".join(ai_spell_issues) + "\n\n") if ai_spell_issues else "Yazım sorunu tespit edilmedi.\n\n"
            res += "--- NLP METRİKLERİ (YAPAY ZEKA BAŞARISI) ---\n"
            res += f"BERTScore Precision : {P.mean().item():.4f}\n"
            res += f"BERTScore Recall    : {R.mean().item():.4f}\n"
            res += f"BERTScore F1-Score  : {F1.mean().item():.4f}\n"
            res += f"Cosine Similarity   : {cos_sim:.4f}\n"
            
            st.session_state.nlp_res = res

if analyze_read_btn:
    if ai_text and ref_text:
        st.session_state.read_res = {
            "ai": calculate_readability_metrics(ai_text),
            "ref": calculate_readability_metrics(ref_text)
        }

# --- SONUÇLARI GÖSTERME (Aynı Anda) ---

st.divider()

col_res1, col_res2 = st.columns(2)

with col_res1:
    st.subheader("📊 NLP Sonuçları")
    if st.session_state.nlp_res:
        st.code(st.session_state.nlp_res, language="text")
    else:
        st.info("NLP analizi için butona basın.")

with col_res2:
    st.subheader("📖 Okunabilirlik Sonuçları")
    if st.session_state.read_res:
        r_ai = st.session_state.read_res["ai"]
        r_ref = st.session_state.read_res["ref"]
        
        st.write(f"**AI Metni (Ateşman):** {r_ai['atesman']} ({get_atesman_desc(r_ai['atesman'])})")
        st.write(f"**Referans (Ateşman):** {r_ref['atesman']} ({get_atesman_desc(r_ref['atesman'])})")
        
        st.table({
            "Metrik": ["Ateşman", "Çetinkaya-Uzun", "Cümle Uz.", "Kelime Uz."],
            "AI": [r_ai['atesman'], r_ai['cetinkaya'], r_ai['avg_sentence'], r_ai['avg_syllable']],
            "Ref": [r_ref['atesman'], r_ref['cetinkaya'], r_ref['avg_sentence'], r_ref['avg_syllable']]
        })
    else:
        st.info("Okunabilirlik analizi için butona basın.")
