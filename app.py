import streamlit as st
import re
import numpy as np
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa Genişliği
st.set_page_config(page_title="Türkçe NLP Analiz Aracı", layout="wide")

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

def calculate_readability(text):
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    if not words or not sentences:
        return None
    
    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(len(re.findall(r'[aeıioöuüAEIİOÖUÜ]', w)) for w in words)
    
    avg_word_len_syllable = total_syllables / total_words
    avg_sentence_len_word = total_words / total_sentences

    # Ateşman ve Çetinkaya-Uzun
    atesman = 199 - (1.015 * avg_sentence_len_word) - (42.332 * avg_word_len_syllable)
    cetinkaya = 118.8 - (25.9 * avg_word_len_syllable) - (0.9 * avg_sentence_len_word)
    
    return atesman, cetinkaya

# --- UI ---
st.title("Türkçe Metin Analiz Aracı (BERTScore & Okunabilirlik)")

ai_text = st.text_area("AI Cümlesi (Aday / Üretilen Metin):", value="Yapay zeka modelleri günümüzde çok hızlı gelişiyor.", height=150)
ref_text = st.text_area("Cevap Anahtarı (Referans Metin):", value="Günümüzde yapay zeka sistemleri oldukça süratli bir gelişim gösteriyor.", height=150)

col1, col2 = st.columns(2)
analyze_btn = col1.button("Analiz Et")
readability_btn = col2.button("Okunabilirlik Hesapla")

if analyze_btn:
    if ai_text and ref_text:
        with st.spinner("Analiz ediliyor..."):
            # Orijinal Mantığın
            ai_words = re.findall(r'\b\w+\b', ai_text, re.UNICODE)
            ref_words = re.findall(r'\b\w+\b', ref_text, re.UNICODE)
            ai_sentences = [s for s in re.split(r'[.!?]+', ai_text) if s.strip()]
            ref_sentences = [s for s in re.split(r'[.!?]+', ref_text) if s.strip()]
            ai_spell_issues = check_turkish_spelling(ai_words)

            P, R, F1 = score([ai_text], [ref_text], lang="tr", verbose=False)
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            embeddings = model.encode([ai_text, ref_text])
            cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            # SENİN İSTEDİĞİN TAM ÇIKTI FORMATI
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

            st.code(res, language="text") # Kod bloğu içinde tam istediğin formatta gösterir

if readability_btn:
    if ai_text and ref_text:
        ai_at, ai_cu = calculate_readability(ai_text)
        ref_at, ref_cu = calculate_readability(ref_text)
        
        read_res = "--- OKUNABİLİRLİK ANALİZİ ---\n"
        read_res += f"[AI Metni] Ateşman: {ai_at:.2f} | Çetinkaya-Uzun: {ai_cu:.2f}\n"
        read_res += f"[Referans] Ateşman: {ref_at:.2f} | Çetinkaya-Uzun: {ref_cu:.2f}\n\n"
        read_res += "--- KARŞILAŞTIRMALI DEĞERLENDİRME ---\n"
        read_res += f"Ateşman Farkı: {abs(ai_at - ref_at):.2f}\n"
        read_res += f"Çetinkaya-Uzun Farkı: {abs(ai_cu - ref_cu):.2f}\n"
        
        st.code(read_res, language="text")
