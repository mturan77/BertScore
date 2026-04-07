# 🇹🇷 Türkçe Metin Analizi ve Okunabilirlik Aracı

Bu proje, yapay zeka tarafından üretilen Türkçe metinlerin kalitesini hem **anlamsal** hem de **yapısal** olarak ölçmek için geliştirilmiştir. Streamlit arayüzü sayesinde kullanıcı dostu bir deneyim sunar.

## 🚀 Özellikler

### 1. NLP Benzerlik Metrikleri
Metinlerin anlam derinliğini ve referans metne yakınlığını ölçer:
* **BERTScore:** Kelime diziliminden ziyade, kelimelerin bağlamsal anlamlarını (BERT modeli kullanarak) karşılaştırır.
* **Cosine Similarity:** Metinleri vektör evrenine taşır ve aralarındaki açısal benzerliği hesaplar.

### 2. Okunabilirlik Analizleri (Readability)
Metnin hedef kitleye uygunluğunu iki farklı formül ile hesaplar:

* **Ateşman Okunabilirlik İndeksi:** * *Formül:* $199 - (1.015 \times \text{Cümle Uzunluğu}) - (42.332 \times \text{Kelime Uzunluğu})$
    * **90-100:** Çok Kolay
    * **70-89:** Kolay
    * **50-69:** Orta Güçlük
    * **30-49:** Zor
    * **0-29:** Çok Zor

* **Çetinkaya-Uzun Formülü:** Türkçe kelime ve cümle yapısına göre optimize edilmiş, yetişkinlerin okuma seviyesini belirleyen güncel bir metriktir.

## 🛠️ Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için:

1. Bu depoyu indirin.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
