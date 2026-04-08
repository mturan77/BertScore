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
    
    # Hece sayımları ve Bezirci-Yılmaz için değişkenler
    total_syllables = 0
    h3, h4, h5, h6 = 0, 0, 0, 0
    
    for w in words:
        syl_count = len(re.findall(r'[aeıioöuüAEIİOÖUÜ]', w))
        total_syllables += syl_count
        # Bezirci-Yılmaz hece gruplandırması
        if syl_count == 3: h3 += 1
        elif syl_count == 4: h4 += 1
        elif syl_count == 5: h5 += 1
        elif syl_count >= 6: h6 += 1

    avg_w_syl = total_syllables / total_words
    avg_s_word = total_words / total_sentences
    
    # Bezirci-Yılmaz Formülü Hesaplaması
    oks = avg_s_word
    avg_h3 = h3 / total_sentences
    avg_h4 = h4 / total_sentences
    avg_h5 = h5 / total_sentences
    avg_h6 = h6 / total_sentences
    
    bezirci_yilmaz = (oks ** 0.5) * ((avg_h3 * 0.84) + (avg_h4 * 1.5) + (avg_h5 * 3.5) + (avg_h6 * 26.35))
    
    atesman = 199 - (1.015 * avg_s_word) - (42.332 * avg_w_syl)
    cetinkaya = 118.8 - (25.9 * avg_w_syl) - (0.9 * avg_s_word)
    
    return {
        "at": round(atesman, 2), 
        "cu": round(cetinkaya, 2), 
        "by": round(bezirci_yilmaz, 2), # Yeni değer eklendi
        "asw": round(avg_s_word, 2), 
        "aws": round(avg_w_syl, 2)
    }

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

# Yeni: Bezirci-Yılmaz Bilgi Fonksiyonu
def get_bezirci_yilmaz_info(score):
    if score is None: return "Veri Yok", "❓"
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

# --- NLP ANALİZ MANTIĞI ---
if c_btn1.button("🚀 NLP Analizi"):
    if ai_text and ref_text:
        with st.spinner("NLP Metrikleri Hesaplanıyor..."):
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
            
            comment = "Anlamsal uyum mükemmel, metinler birbirine çok yakın." if f1_val > 0.88 else "Anlamsal olarak metinler arasında belirgin farklar mevcut."
            st.session_state.nlp_res = {"raw": raw_report, "f1": f1_val, "cos": cos_sim, "comment": comment}

# --- OKUNABİLİRLİK MANTIĞI ---
if c_btn2.button("📊 Okunabilirlik Analizi"):
    if ai_text and ref_text:
        r_ai, r_ref = calculate_readability(ai_text), calculate_readability(ref_text)
        diff = abs(r_ai['at'] - r_ref['at'])
        comment = "AI metni, referansın dil ağırlığına tam uyum sağlıyor." if diff < 10 else "AI metni referanstan farklı bir karmaşıklık düzeyinde."
        st.session_state.read_res = {"ai": r_ai, "ref": r_ref, "comment": comment}

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
        st.success(f"**💡 NLP Analist Yorumu:** {n.get('comment')}")
    else:
        st.info("NLP analizi bekleniyor...")

with res_right:
    st.subheader("📈 Okunabilirlik Dashboard")
    if st.session_state.read_res and isinstance(st.session_state.read_res, dict):
        r = st.session_state.read_res
        ai_d, ref_d = r.get("ai", {}), r.get("ref", {})

        # Ateşman Bölümü
        st.markdown("**Ateşman İndeksi (Genel Zorluk)**")
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
        m4.metric("Ref Çetinkaya", f"{ref_d.get('cu')}", f"{cl_ref} {ci_ref}", delta_color="off")

        # Yeni: Bezirci-Yılmaz Bölümü
        st.markdown("**Bezirci-Yılmaz (Eğitim Seviyesi)**")
        m5, m6 = st.columns(2)
        by_l_ai, by_i_ai = get_bezirci_yilmaz_info(ai_d.get("by"))
        by_l_ref, by_i_ref = get_bezirci_yilmaz_info(ref_d.get("by"))
        m5.metric("AI Bezirci-Yılmaz", f"{ai_d.get('by')}", f"{by_l_ai} {by_i_ai}", delta_color="off")
        m6.metric("Ref Bezirci-Yılmaz", f"{ref_d.get('by')}", f"{by_l_ref} {by_i_ref}", delta_color="off")

        # Tablo ve Yorum (Tabloya Bezirci-Yılmaz eklendi)
        st.table({
            "Ölçüt": ["Ateşman Puanı", "Çetinkaya-Uzun", "Bezirci-Yılmaz", "Ort. Kelime Sayısı", "Ort. Hece Sayısı"],
            "AI Metni": [ai_d.get("at"), ai_d.get("cu"), ai_d.get("by"), ai_d.get("asw"), ai_d.get("aws")],
            "Referans": [ref_d.get("at"), ref_d.get("cu"), ref_d.get("by"), ref_d.get("asw"), ref_d.get("aws")]
        })
        st.info(f"**💡 Okunabilirlik Yorumu:** {r.get('comment')}")
    else:
        st.info("Okunabilirlik analizi bekleniyor...")
