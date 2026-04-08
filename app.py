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
    total_syllables = 0
    h3, h4, h5, h6 = 0, 0, 0, 0
    
    for w in words:
        syl_count = len(re.findall(r'[aeıioöuüAEIİOÖUÜ]', w))
        total_syllables += syl_count
        if syl_count == 3: h3 += 1
        elif syl_count == 4: h4 += 1
        elif syl_count == 5: h5 += 1
        elif syl_count >= 6: h6 += 1

    # Temel Metrikler
    oks = total_words / total_sentences
    ohs = total_syllables / total_words
    
    # Bezirci-Yılmaz Ara Değerleri
    avg_h3, avg_h4, avg_h5, avg_h6 = h3/total_sentences, h4/total_sentences, h5/total_sentences, h6/total_sentences
    
    # Formüller
    atesman = 199 - (1.015 * oks) - (42.332 * ohs)
    cetinkaya = 118.8 - (25.9 * ohs) - (0.9 * oks)
    bezirci_yilmaz = (oks ** 0.5) * ((avg_h3 * 0.84) + (avg_h4 * 1.5) + (avg_h5 * 3.5) + (avg_h6 * 26.35))
    
    return {
        "raw": {"kelime": total_words, "cumle": total_sentences, "hece": total_syllables, "h3": h3, "h4": h4, "h5": h5, "h6": h6},
        "averages": {"oks": round(oks, 3), "ohs": round(ohs, 3), "h3_avg": round(avg_h3, 3), "h4_avg": round(avg_h4, 3), "h5_avg": round(avg_h5, 3), "h6_avg": round(avg_h6, 3)},
        "scores": {"at": round(atesman, 2), "cu": round(cetinkaya, 2), "by": round(bezirci_yilmaz, 2)}
    }

def get_atesman_info(score):
    if score >= 90: return "Çok Kolay", "🍀"
    if score >= 70: return "Kolay", "✅"
    if score >= 50: return "Orta Güçlük", "⚠️"
    if score >= 30: return "Zor", "🚩"
    return "Çok Zor", "🚫"

def get_cetinkaya_info(score):
    if score >= 80: return "Eğitim düzeyi düşük", "📗"
    if score >= 60: return "Orta eğitim düzeyi", "📘"
    if score >= 40: return "Yüksek eğitim düzeyi", "📙"
    return "Akademik/Uzman düzey", "📕"

def get_bezirci_yilmaz_info(score):
    if score < 9: return "İlköğretim", "🟢"
    if score < 13: return "Lise", "🟡"
    if score <= 16: return "Lisans", "🟠"
    return "Akademik", "🔴"

# --- ARAYÜZ ---
st.title("🇹🇷 Gelişmiş Türkçe Metin Analiz Dashboard")

col_in1, col_in2 = st.columns(2)
ai_text = col_in1.text_area("AI Metni (Aday):", value="Yapay zeka modelleri günümüzde çok hızlı gelişiyor.", height=150)
ref_text = col_in2.text_area("Referans Metin (Cevap Anahtarı):", value="Günümüzde yapay zeka sistemleri oldukça süratli bir gelişim gösteriyor.", height=150)

c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 4])

if c_btn1.button("🚀 NLP Analizi"):
    if ai_text and ref_text:
        with st.spinner("NLP Metrikleri Hesaplanıyor..."):
            ai_words = re.findall(r'\b\w+\b', ai_text, re.UNICODE)
            ref_words = re.findall(r'\b\w+\b', ref_text, re.UNICODE)
            P, R, F1 = score([ai_text], [ref_text], lang="tr", verbose=False)
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            cos_sim = cosine_similarity([model.encode(ai_text)], [model.encode(ref_text)])[0][0]
            f1_val = F1.mean().item()
            st.session_state.nlp_res = {"f1": f1_val, "cos": cos_sim, "report": f"AI: {len(ai_words)} Kelime | Ref: {len(ref_words)} Kelime"}

if c_btn2.button("📊 Okunabilirlik Analizi"):
    if ai_text and ref_text:
        st.session_state.read_res = {"ai": calculate_readability(ai_text), "ref": calculate_readability(ref_text)}

if c_btn3.button("🗑️ Temizle"):
    st.session_state.nlp_res = st.session_state.read_res = None
    st.rerun()

st.divider()

if st.session_state.read_res:
    r = st.session_state.read_res
    
    st.header("📈 Okunabilirlik ve Adım Adım Çözüm")
    
    # Üst Metrik Kartları
    m1, m2, m3 = st.columns(3)
    with m1:
        score = r['ai']['scores']['at']
        lbl, icon = get_atesman_info(score)
        st.metric("Ateşman İndeksi (AI)", score, f"{lbl} {icon}")
    with m2:
        score = r['ai']['scores']['cu']
        lbl, icon = get_cetinkaya_info(score)
        st.metric("Çetinkaya-Uzun (AI)", score, f"{lbl} {icon}")
    with m3:
        score = r['ai']['scores']['by']
        lbl, icon = get_bezirci_yilmaz_info(score)
        st.metric("Bezirci-Yılmaz (AI)", score, f"{lbl} {icon}")

    st.subheader("📝 Fizik Sınavı Formatında Çözüm (AI Metni İçin)")
    tab1, tab2, tab3 = st.tabs(["1. Ateşman Çözümü", "2. Çetinkaya-Uzun Çözümü", "3. Bezirci-Yılmaz Çözümü"])
    
    data = r['ai']
    
    with tab1:
        st.markdown("### **Ateşman Okunabilirlik Formülü**")
        st.info(f"**Verilenler:** OKS (Ort. Kelime): {data['averages']['oks']} | OHS (Ort. Hece): {data['averages']['ohs']}")
        st.latex(r"Formül: 199 - (1.015 \times OKS) - (42.332 \times OHS)")
        st.latex(rf"Yerine \ Koyma: 199 - (1.015 \times {data['averages']['oks']}) - (42.332 \times {data['averages']['ohs']})")
        st.success(rf"**Sonuç:** {data['scores']['at']} ({get_atesman_info(data['scores']['at'])[0]})")

    with tab2:
        st.markdown("### **Çetinkaya-Uzun Okunabilirlik Formülü**")
        st.info(f"**Verilenler:** OKS: {data['averages']['oks']} | OHS: {data['averages']['ohs']}")
        st.latex(r"Formül: 118.8 - (25.9 \times OHS) - (0.9 \times OKS)")
        st.latex(rf"Yerine \ Koyma: 118.8 - (25.9 \times {data['averages']['ohs']}) - (0.9 \times {data['averages']['oks']})")
        st.success(rf"**Sonuç:** {data['scores']['cu']} ({get_cetinkaya_info(data['scores']['cu'])[0]})")

    with tab3:
        st.markdown("### **Bezirci-Yılmaz Okunabilirlik Formülü**")
        st.info(f"**Verilenler:** OKS: {data['averages']['oks']} | H3: {data['averages']['h3_avg']} | H4: {data['averages']['h4_avg']} | H5: {data['averages']['h5_avg']} | H6+: {data['averages']['h6_avg']}")
        st.latex(r"Formül: \sqrt{OKS} \times [(H3 \times 0.84) + (H4 \times 1.5) + (H5 \times 3.5) + (H6 \times 26.35)]")
        st.latex(rf"Yerine \ Koyma: \sqrt{{{data['averages']['oks']}}} \times [({data['averages']['h3_avg']} \times 0.84) + ({data['averages']['h4_avg']} \times 1.5) + ({data['averages']['h5_avg']} \times 3.5) + ({data['averages']['h6_avg']} \times 26.35)]")
        st.success(rf"**Sonuç:** {data['scores']['by']} ({get_bezirci_yilmaz_info(data['scores']['by'])[0]})")

    # Genel Karşılaştırma Tablosu
    st.subheader("📊 Karşılaştırmalı Özet Tablo")
    st.table({
        "Metrik": ["Ateşman", "Çetinkaya-Uzun", "Bezirci-Yılmaz", "Ort. Kelime (OKS)", "Ort. Hece (OHS)"],
        "AI Metni": [r['ai']['scores']['at'], r['ai']['scores']['cu'], r['ai']['scores']['by'], r['ai']['averages']['oks'], r['ai']['averages']['ohs']],
        "Referans": [r['ref']['scores']['at'], r['ref']['scores']['cu'], r['ref']['scores']['by'], r['ref']['averages']['oks'], r['ref']['averages']['ohs']]
    })
