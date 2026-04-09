import streamlit as st
import re
import numpy as np
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Türkçe Metin Analiz Dashboard", layout="wide")

# --- Session State Başlatma ---
if 'nlp_res' not in st.session_state: st.session_state.nlp_res = None
if 'read_res' not in st.session_state: st.session_state.read_res = None

# --- NLP MODEL CACHE FONKSİYONLARI ---
@st.cache_resource
def get_bert_scorer():
    # rescale_with_baseline=False yapıldı çünkü Türkçe (lang="tr") için baseline dosyası yok.
    return BERTScorer(lang="tr", rescale_with_baseline=False)

@st.cache_resource
def get_sentence_transformer():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- FONKSİYONLAR ---
def check_turkish_spelling(words):
    kalin, ince = set("aıou"), set("eiöü")
    return [w for w in words if any(c in kalin for c in w.lower()) and any(c in ince for c in w.lower())]

def calculate_readability(text):
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    
    words_raw = re.findall(r'\b\w+\b', text, re.UNICODE)
    words = [w for w in words_raw if any(c.isalpha() for c in w)]
    
    if not words or not sentences: 
        return None
    
    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = 0
    h3, h4, h5, h6 = 0, 0, 0, 0
    
    for w in words:
        syl_count = len(re.findall(r'[aeıioöuüAEIİOÖUÜ]', w))
        if syl_count == 0:
            syl_count = 1 
            
        total_syllables += syl_count
        
        if syl_count == 3: h3 += 1
        elif syl_count == 4: h4 += 1
        elif syl_count == 5: h5 += 1
        elif syl_count >= 6: h6 += 1

    oks = total_words / total_sentences
    ohs = total_syllables / total_words
    
    avg_h3 = h3 / total_sentences
    avg_h4 = h4 / total_sentences
    avg_h5 = h5 / total_sentences
    avg_h6 = h6 / total_sentences
    
    # 1. Ateşman Formülü
    atesman = 198.825 - (40.175 * ohs) - (2.610 * oks)
    
    # 2. Çetinkaya-Uzun Formülü
    cetinkaya = 118.823 - (25.987 * ohs) - (0.971 * oks)
    
    # 3. Bezirci-Yılmaz Formülü (Karekök eklendi)
    bezirci_yilmaz = (oks ** 0.5) * ((avg_h3 * 0.84) + (avg_h4 * 1.5) + (avg_h5 * 3.5) + (avg_h6 * 26.25))
    
    return {
        "raw": {"kelime": total_words, "cumle": total_sentences, "hece": total_syllables},
        "averages": {
            "oks": round(oks, 3), "ohs": round(ohs, 3), 
            "h3": round(avg_h3, 3), "h4": round(avg_h4, 3), 
            "h5": round(avg_h5, 3), "h6": round(avg_h6, 3)
        },
        "scores": {
            "at": round(atesman, 2), "cu": round(cetinkaya, 2), "by": round(bezirci_yilmaz, 2)
        }
    }

def get_atesman_info(score):
    if score >= 90: return "Çok Kolay", "🍀"
    if score >= 70: return "Kolay", "✅"
    if score >= 50: return "Orta Güçlükte", "⚠️"
    if score >= 30: return "Zor", "🚩"
    return "Çok Zor", "🚫"

def get_cetinkaya_info(score):
    if score >= 80: return "Kolay (6. Sınıf)", "📗"
    if score >= 70: return "Biraz Kolay (7. Sınıf)", "📘"
    if score >= 60: return "Standart (8-9. Sınıf)", "📙"
    if score >= 50: return "Biraz Zor (10-12. Sınıf)", "📕"
    if score >= 30: return "Zor (Lisans)", "🏛️"
    return "Çok Zor (Akademik)", "🎓"

def get_bezirci_yilmaz_info(score):
    if score <= 8: return "İlköğretim", "🟢"     
    if score <= 12: return "Lise", "🟡"          
    if score <= 16: return "Lisans", "🟠"        
    return "Akademik", "🔴"                      

def render_exam_solution(data, title):
    m1, m2, m3 = st.columns(3)
    with m1:
        score = data['scores']['at']
        lbl, icon = get_atesman_info(score)
        st.metric(f"Ateşman ({title})", score, f"{lbl} {icon}")
    with m2:
        score = data['scores']['cu']
        lbl, icon = get_cetinkaya_info(score)
        st.metric(f"Çetinkaya-Uzun ({title})", score, f"{lbl} {icon}")
    with m3:
        score = data['scores']['by']
        lbl, icon = get_bezirci_yilmaz_info(score)
        st.metric(f"Bezirci-Yılmaz ({title})", score, f"{lbl} {icon}")

    st.markdown("---")
    st.subheader(f"📝 Fizik Sınavı Çözüm Kâğıdı ({title})")
    
    t1, t2, t3 = st.tabs(["1. Ateşman Çözümü", "2. Çetinkaya-Uzun Çözümü", "3. Bezirci-Yılmaz Çözümü"])
    
    oks = data['averages']['oks']
    ohs = data['averages']['ohs']
    
    with t1:
        st.markdown("**1. Verilenler**")
        st.code(f"OKS (Ort. Kelime Sayısı) = {oks}\nOHS (Ort. Hece Sayısı) = {ohs}", language="text")
        st.markdown("**2. Formül**")
        st.latex(r"198.825 - (40.175 \times OHS) - (2.610 \times OKS)")
        st.markdown("**3. Yerine Koyma**")
        st.latex(rf"198.825 - (40.175 \times {ohs}) - (2.610 \times {oks})")
        st.markdown("**4. Ara Çarpmalar**")
        c1, c2 = round(40.175 * ohs, 3), round(2.610 * oks, 3)
        st.latex(rf"198.825 - {c1} - {c2}")
        st.markdown("**5. Sonuç**")
        st.success(f"Skor: {data['scores']['at']}  |  Yorum: {get_atesman_info(data['scores']['at'])[0]}")

    with t2:
        st.markdown("**1. Verilenler**")
        st.code(f"OKS = {oks}\nOHS = {ohs}", language="text")
        st.markdown("**2. Formül**")
        st.latex(r"118.823 - (25.987 \times OHS) - (0.971 \times OKS)")
        st.markdown("**3. Yerine Koyma**")
        st.latex(rf"118.823 - (25.987 \times {ohs}) - (0.971 \times {oks})")
        st.markdown("**4. Ara Çarpmalar**")
        c1, c2 = round(25.987 * ohs, 3), round(0.971 * oks, 3)
        st.latex(rf"118.823 - {c1} - {c2}")
        st.markdown("**5. Sonuç**")
        st.success(f"Skor: {data['scores']['cu']}  |  Yorum: {get_cetinkaya_info(data['scores']['cu'])[0]}")

    with t3:
        h3, h4, h5, h6 = data['averages']['h3'], data['averages']['h4'], data['averages']['h5'], data['averages']['h6']
        st.markdown("**1. Verilenler**")
        st.code(f"OKS = {oks}\nH3 (3 Heceli Ort) = {h3}\nH4 (4 Heceli Ort) = {h4}\nH5 (5 Heceli Ort) = {h5}\nH6+ (6+ Heceli Ort) = {h6}", language="text")
        st.markdown("**2. Formül**")
        st.latex(r"\sqrt{OKS} \times [(H_3 \times 0.84) + (H_4 \times 1.5) + (H_5 \times 3.5) + (H_6 \times 26.25)]")
        st.markdown("**3. Yerine Koyma**")
        st.latex(rf"\sqrt{{{oks}}} \times [({h3} \times 0.84) + ({h4} \times 1.5) + ({h5} \times 3.5) + ({h6} \times 26.25)]")
        st.markdown("**4. Karekök ve Parantez İçi Çarpmalar**")
        kok = round(oks**0.5, 3)
        c3, c4, c5, c6 = round(h3*0.84, 3), round(h4*1.5, 3), round(h5*3.5, 3), round(h6*26.25, 3)
        st.latex(rf"{kok} \times [{c3} + {c4} + {c5} + {c6}]")
        st.markdown("**5. Toplam ve Son Çarpım**")
        toplam = round(c3 + c4 + c5 + c6, 3)
        st.latex(rf"{kok} \times {toplam}")
        st.markdown("**6. Sonuç**")
        st.success(f"Skor: {data['scores']['by']}  |  Yorum: {get_bezirci_yilmaz_info(data['scores']['by'])[0]}")

# --- ANA UYGULAMA ---
st.title("🇹🇷 Gelişmiş Türkçe Metin Analiz Dashboard")

col_in1, col_in2 = st.columns(2)
ai_text = col_in1.text_area("AI Metni (Aday):", value="Yapay zeka modelleri günümüzde çok hızlı gelişiyor.", height=150)
ref_text = col_in2.text_area("Referans Metin (Cevap Anahtarı):", value="Günümüzde yapay zeka sistemleri oldukça süratli bir gelişim gösteriyor.", height=150)

c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 4])

# --- NLP ANALİZİ ---
if c_btn1.button("🚀 NLP Analizi"):
    if ai_text and ref_text:
        with st.spinner("NLP Metrikleri Hesaplanıyor... (Modellerin ilk yüklenmesi biraz sürebilir)"):
            ai_words = re.findall(r'\b\w+\b', ai_text, re.UNICODE)
            ref_words = re.findall(r'\b\w+\b', ref_text, re.UNICODE)
            ai_sentences = [s for s in re.split(r'[.!?]+', ai_text) if s.strip()]
            ref_sentences = [s for s in re.split(r'[.!?]+', ref_text) if s.strip()]
            
            ai_spell_issues = check_turkish_spelling(ai_words)
            
            # Cachelenmiş modelleri çağırıyoruz
            scorer = get_bert_scorer()
            P, R, F1 = scorer.score([ai_text], [ref_text])
            f1_val = F1.mean().item()
            
            model = get_sentence_transformer()
            cos_sim = cosine_similarity([model.encode(ai_text)], [model.encode(ref_text)])[0][0]
            
            raw_report = f"--- TEMEL ÖZELLİKLER ---\n[AI] Kelime: {len(ai_words)} | Cümle: {len(ai_sentences)}\n"
            raw_report += f"[Ref] Kelime: {len(ref_words)} | Cümle: {len(ref_sentences)}\n\n"
            raw_report += "--- İMLA KONTROLÜ (AI) ---\n"
            raw_report += (", ".join(ai_spell_issues) + "\n\n") if ai_spell_issues else "Sorun yok.\n\n"
            raw_report += f"--- TEKNİK METRİKLER ---\nPrecision : {P.mean().item():.4f}\nRecall    : {R.mean().item():.4f}\nF1-Score  : {f1_val:.4f}\nCosine    : {cos_sim:.4f}"
            
            # Yorum eşiği eski haline (0.88) döndürüldü
            comment = "Anlamsal uyum mükemmel, metinler birbirine çok yakın." if f1_val > 0.88 else "Anlamsal olarak metinler arasında belirgin farklar mevcut."
            st.session_state.nlp_res = {"raw": raw_report, "f1": f1_val, "cos": cos_sim, "comment": comment}

# --- OKUNABİLİRLİK ANALİZİ ---
if c_btn2.button("📊 Okunabilirlik Analizi"):
    if ai_text and ref_text:
        st.session_state.read_res = {
            "ai": calculate_readability(ai_text), 
            "ref": calculate_readability(ref_text)
        }

if c_btn3.button("🗑️ Temizle"):
    st.session_state.nlp_res = st.session_state.read_res = None
    st.rerun()

st.divider()

# --- SONUÇLARIN GÖSTERİMİ ---

# 1. NLP Analiz Sonuçları
if st.session_state.nlp_res:
    st.header("🛠️ NLP Analiz Raporu")
    n = st.session_state.nlp_res
    
    n_col1, n_col2 = st.columns([1, 2])
    with n_col1:
        st.metric("F1-Score (Başarı)", f"{n.get('f1'):.2%}")
        st.metric("Anlam Benzerliği (Cosine)", f"{n.get('cos'):.2%}")
    with n_col2:
        st.code(n.get('raw'), language="text")
        
    st.success(f"**💡 NLP Analist Yorumu:** {n.get('comment')}")
    st.divider()

# 2. Okunabilirlik Analiz Sonuçları
if st.session_state.read_res:
    r = st.session_state.read_res
    st.header("📈 Okunabilirlik ve Adım Adım Çözüm")
    
    main_tab_ai, main_tab_ref = st.tabs(["🤖 AI Metni Analizi", "📝 Referans Metin Analizi"])
    
    with main_tab_ai:
        render_exam_solution(r['ai'], "AI")
        
    with main_tab_ref:
        render_exam_solution(r['ref'], "Ref")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("📊 Karşılaştırmalı Özet Tablo")
    st.table({
        "Metrik": ["Ateşman Puanı", "Çetinkaya-Uzun Puanı", "Bezirci-Yılmaz Puanı", "Ort. Kelime (OKS)", "Ort. Hece (OHS)"],
        "AI Metni": [r['ai']['scores']['at'], r['ai']['scores']['cu'], r['ai']['scores']['by'], r['ai']['averages']['oks'], r['ai']['averages']['ohs']],
        "Referans": [r['ref']['scores']['at'], r['ref']['scores']['cu'], r['ref']['scores']['by'], r['ref']['averages']['oks'], r['ref']['averages']['ohs']]
    })
