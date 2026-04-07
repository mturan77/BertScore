import streamlit as st
import re
import numpy as np
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa Ayarları
st.set_page_config(page_title="Türkçe NLP ve Okunabilirlik Analizi", layout="wide")

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
    if not words or not sentences:
        return None
    
    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(len(re.findall(r'[aeıioöuüAEIİOÖUÜ]', w)) for w in words)
    
    avg_word_len_syllable = total_syllables / total_words
    avg_sentence_len_word = total_words / total_sentences

    atesman = 199 - (1.015 * avg_sentence_len_word) - (42.332 * avg_word_len_syllable)
    cetinkaya = 118.8 - (25.9 * avg_word_len_syllable) - (0.9 * avg_sentence_len_word)
    
    return {
        "atesman": round(atesman, 2),
        "cetinkaya": round(cetinkaya, 2),
        "avg_sentence": round(avg_sentence_len_word, 2),
        "avg_syllable": round(avg_word_len_syllable, 2)
    }

def get_atesman_desc(score):
    if score >= 90: return "Çok Kolay (90-100)"
    if score >= 70: return "Kolay (70-89)"
    if score >= 50: return "Orta Güçlük (50-69)"
    if score >= 30: return "Zor (30-49)"
    return "Çok Zor (0-29)"

# --- ARAYÜZ ---
st.title("Türkçe Metin Analiz Aracı")

ai_text = st.text_area("AI Cümlesi (Aday / Üretilen Metin):", value="Yapay zeka modelleri günümüzde çok hızlı gelişiyor.", height=150)
ref_text = st.text_area("Cevap Anahtarı (Referans Metin):", value="Günümüzde yapay zeka sistemleri oldukça süratli bir gelişim gösteriyor.", height=150)

col1, col2 = st.columns(2)
analyze_btn = col1.button("NLP Benzerlik Analizi")
readability_btn = col2.button("Okunabilirlik Analizi")

# --- 1. NLP ANALİZİ (İSTEDİĞİN HAM METİN FORMATI) ---
if analyze_btn:
    if ai_text and ref_text:
        with st.spinner("Analiz ediliyor..."):
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
            if ai_spell_issues:
                res += "Dikkat - Büyük Ünlü Uyumuna uymayan (Yabancı kökenli/Hatalı) kelimeler:\n"
                res += ", ".join(ai_spell_issues) + "\n\n"
            else:
                res += "Yazım sorunu veya ünlü uyumu ihlali tespit edilmedi.\n\n"

            res += "--- NLP METRİKLERİ (YAPAY ZEKA BAŞARISI) ---\n"
            res += f"BERTScore Precision (Hassasiyet) : {P.mean().item():.4f}\n"
            res += f"BERTScore Recall (Duyarlılık)    : {R.mean().item():.4f}\n"
            res += f"BERTScore F1-Score (Genel Başarı): {F1.mean().item():.4f}\n"
            res += f"Cosine Similarity (Anlam Benzerliği): {cos_sim:.4f}\n"

            st.code(res, language="text")

# --- 2. OKUNABİLİRLİK (DETAYLI VE GÖRSEL FORMAT) ---
if readability_btn:
    if ai_text and ref_text:
        res_ai = calculate_readability_metrics(ai_text)
        res_ref = calculate_readability_metrics(ref_text)

        st.subheader("📖 Okunabilirlik Karşılaştırması")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**AI Metni (Ateşman):** {res_ai['atesman']}\n\n**Seviye:** {get_atesman_desc(res_ai['atesman'])}")
        with c2:
            st.success(f"**Referans (Ateşman):** {res_ref['atesman']}\n\n**Seviye:** {get_atesman_desc(res_ref['atesman'])}")
        
        st.markdown("---")
        
        st.write("**Metrik Karşılaştırma Tablosu:**")
        st.table({
            "Metrik": ["Ateşman Puanı", "Çetinkaya-Uzun Puanı", "Ort. Cümle (Kelime)", "Ort. Kelime (Hece)"],
            "AI Metni": [res_ai['atesman'], res_ai['cetinkaya'], res_ai['avg_sentence'], res_ai['avg_syllable']],
            "Referans Metin": [res_ref['atesman'], res_ref['cetinkaya'], res_ref['avg_sentence'], res_ref['avg_syllable']]
        })
        
        st.write(f"**Yorum:** İki metin arasındaki Ateşman farkı **{round(abs(res_ai['atesman'] - res_ref['atesman']), 2)}** puandır.")
