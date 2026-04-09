import re
import streamlit as st
from bert_score import BERTScorer

# --- NLP MODEL CACHE FONKSİYONU (YENİ EKLENDİ) ---
@st.cache_resource
def get_bert_scorer():
    """
    BERT modelinin butona her basıldığında yeniden yüklenmesini önlemek
    ve bellek sızıntılarını engellemek için model önbelleğe (cache) alınır.
    rescale_with_baseline=True parametresi, daralan puan aralığını
    insan tarafından yorumlanabilir gerçekçi bir spektruma yayar.
    """
    # lang="tr" varsayılan olarak 'dbmdz/bert-base-turkish-cased' modelini kullanır.
    return BERTScorer(lang="tr", rescale_with_baseline=True)

# --- OKUNABİLİRLİK HESAPLAMA FONKSİYONU ---
def calculate_readability(text):
    """
    Türkçe metinler için Ateşman (1997), Çetinkaya-Uzun (2010) ve Bezirci-Yılmaz (2010)
    okunabilirlik formüllerini literatüre eksiksiz uygun olarak hesaplar.
    """
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    
    # Rakamların okunabilirlik OHS metriklerini asimetrik düşürmesini engellemek için filtreleme.
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
    
    # 3. Bezirci-Yılmaz Formülü
    bezirci_yilmaz = oks * ((avg_h3 * 0.84) + (avg_h4 * 1.5) + (avg_h5 * 3.5) + (avg_h6 * 26.25))
    
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
    if score >= 60: return "Standard (8-9. Sınıf)", "📙"
    if score >= 50: return "Biraz Zor (10-12. Sınıf)", "📕"
    if score >= 30: return "Zor (Lisans)", "🏛️"
    return "Çok Zor (Akademik)", "🎓"

def get_bezirci_yilmaz_info(score):
    if score <= 8: return "İlköğretim", "🟢"     
    if score <= 12: return "Lise", "🟡"          
    if score <= 16: return "Lisans", "🟠"        
    return "Akademik", "🔴"                      

# --- NLP ANALİZ BUTONU İÇİ KULLANIM ÖRNEĞİ ---
# Aşağıdaki yapı Streamlit buton bloğunuzun içindeki ilgili yere entegre edilmelidir:
"""
scorer = get_bert_scorer()
P, R, F1 = scorer.score([ai_text], [ref_text])
f1_val = F1.mean().item()

# Rescale edilmiş değerler 0 civarında dalgalanabilir.
# Eşik değeri artık 0.88 değil, ölçeklendirilmiş dağılıma uygun bir değer olmalıdır (örneğin 0.50).
comment = "Anlamsal uyum mükemmel, metinler birbirine oldukça yakın." if f1_val > 0.50 else "Anlamsal olarak metinler arasında belirgin farklar mevcut."
"""
