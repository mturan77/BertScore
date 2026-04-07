import streamlit as st
import re
import numpy as np
import threading
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa Ayarları
st.set_page_config(page_title="Türkçe NLP ve Okunabilirlik Analizi", layout="wide")

# --- Fonksiyonlar ---

def calculate_readability(text):
    """Ateşman ve Çetinkaya-Uzun Okunabilirlik Formülleri"""
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    
    if not words or not sentences:
        return None

    total_words = len(words)
    total_sentences = len(sentences)
    
    # Hece sayımı (Türkçe için sesli harf sayımı pratik bir yaklaşımdır)
    def count_syllables(word):
        return len(re.findall(r'[aeıioöuüAEIİOÖUÜ]', word))

    total_syllables = sum(count_syllables(w) for w in words)
    avg_word_len_syllable = total_syllables / total_words
    avg_sentence_len_word = total_words / total_sentences

    # Ateşman Formülü: 199 - (1.015 * x1) - (42.332 * x2)
    # x1: Kelime olarak ortalama cümle uzunluğu
    # x2: Hece olarak ortalama kelime uzunluğu
    atesman_score = 199 - (1.015 * avg_sentence_len_word) - (42.332 * avg_word_len_syllable)

    # Çetinkaya-Uzun Formülü (Yetişkinler için):
    # Skor = 118.8 - (25.9 * ortalama kelime uzunluğu (hece)) - (0.9 * ortalama cümle uzunluğu (kelime))
    cetinkaya_score = 118.8 - (25.9 * avg_word_len_syllable) - (0.9 * avg_sentence_len_word)

    return {
        "atesman": round(atesman_score, 2),
        "cetinkaya": round(cetinkaya_score, 2),
        "avg_sentence": round(avg_sentence_len_word, 2),
        "avg_syllable": round(avg_word_len_syllable, 2)
    }

def get_atesman_desc(score):
    if score >= 90: return "Çok Kolay (90-100)"
    if score >= 70: return "Kolay (70-89)"
    if score >= 50: return "Orta Güçlük (50-69)"
    if score >= 30: return "Zor (30-49)"
    return "Çok Zor (0-29)"

# --- Streamlit UI ---

st.title("🇹🇷 Türkçe Metin Analizi ve Karşılaştırma")
st.markdown("""
Bu araç, iki metin arasındaki anlamsal benzerliği (BERTScore & Cosine) ölçer ve 
metinlerin okunabilirlik seviyelerini (Ateşman & Çetinkaya-Uzun) hesaplar.
""")

col1, col2 = st.columns(2)

with col1:
    ai_text = st.text_area("AI Cümlesi (Aday Metin):", value="Yapay zeka modelleri günümüzde çok hızlı gelişiyor.", height=200)

with col2:
    ref_text = st.text_area("Cevap Anahtarı (Referans Metin):", value="Günümüzde yapay zeka sistemleri oldukça süratli bir gelişim gösteriyor.", height=200)

# Analiz Butonları
btn_col1, btn_col2 = st.columns(2)
analyze_nlp = btn_col1.button("NLP Benzerlik Analizi Yap")
analyze_readability = btn_col2.button("Okunabilirlik Analizi Yap")

# --- 1. NLP Benzerlik Sonuçları ---
if analyze_nlp:
    if ai_text and ref_text:
        with st.spinner("Modeller yükleniyor ve hesaplanıyor..."):
            # BERTScore
            P, R, F1 = score([ai_text], [ref_text], lang="tr", verbose=False)
            
            # Cosine Similarity
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            embeddings = model.encode([ai_text, ref_text])
            cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            st.subheader("📊 NLP Metrikleri")
            m1, m2, m3 = st.columns(3)
            m1.metric("BERTScore F1", f"{F1.mean().item():.4f}")
            m2.metric("Cosine Similarity", f"{cos_sim:.4f}")
            m3.metric("Precision / Recall", f"{P.mean().item():.2f} / {R.mean().item():.2f}")
    else:
        st.warning("Lütfen her iki metni de doldurun.")

# --- 2. Okunabilirlik Sonuçları ---
if analyze_readability:
    if ai_text and ref_text:
        res_ai = calculate_readability(ai_text)
        res_ref = calculate_readability(ref_text)

        st.subheader("📖 Okunabilirlik Karşılaştırması")
        
        tab1, tab2 = st.tabs(["Ateşman İndeksi", "Çetinkaya-Uzun İndeksi"])
        
        with tab1:
            c1, c2 = st.columns(2)
            c1.info(f"**AI Metni:** {res_ai['atesman']} \n\n Seviye: {get_atesman_desc(res_ai['atesman'])}")
            c2.success(f"**Referans Metin:** {res_ref['atesman']} \n\n Seviye: {get_atesman_desc(res_ref['atesman'])}")
            st.caption("Ateşman skalasında puan yükseldikçe metin kolaylaşır.")

        with tab2:
            c1, c2 = st.columns(2)
            c1.info(f"**AI Metni:** {res_ai['cetinkaya']} Puan")
            c2.success(f"**Referans Metin:** {res_ref['cetinkaya']} Puan")
            st.write(f"**Fark:** {round(abs(res_ai['cetinkaya'] - res_ref['cetinkaya']), 2)} puan")

        st.table({
            "Metrik": ["Ort. Cümle Uzunluğu (Kelime)", "Ort. Kelime Uzunluğu (Hece)"],
            "AI Metni": [res_ai['avg_sentence'], res_ai['avg_syllable']],
            "Referans Metin": [res_ref['avg_sentence'], res_ref['avg_syllable']]
        })
    else:
        st.warning("Okunabilirlik analizi için metin girişi gereklidir.")
