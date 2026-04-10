# main.py
import threading
import customtkinter as ctk

# --- YENİ OLUŞTURDUĞUMUZ MODÜLLERİ ÇEKİYORUZ ---
from module_nlp import process_nlp
from module_readability import process_readability
from ui_components import build_nlp_dashboard, build_readability_gui

# --- TEMA VE GÖRSEL AYARLAR ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TextAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🎓 Yüksek Lisans Tezi - Gelişmiş Metin Analiz Dashboard")
        self.geometry("1450x950")
        self.configure(fg_color="#121212")
        
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        self.main_font = ctk.CTkFont(family="Segoe UI", size=14)
        self.header_font = ctk.CTkFont(family="Segoe UI", size=28, weight="bold")
        self.console_font = ctk.CTkFont(family="Consolas", size=13)

        self.setup_ui()

    def setup_ui(self):
        # 1. Başlık Alanı
        self.header_frame = ctk.CTkFrame(self, fg_color="#1C1C1E", corner_radius=0)
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        ctk.CTkLabel(self.header_frame, text="Yapay Zeka Prompt Analiz Merkezi", font=self.header_font, text_color="#FFFFFF").pack(pady=(20, 5))
        ctk.CTkLabel(self.header_frame, text="Zero-shot, Few-shot ve Chain-of-Thought Çıktı Karşılaştırması", font=self.main_font, text_color="#A0A0A5").pack(pady=(0, 20))

        # 2. Girdi Alanları
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20, pady=15)
        input_frame.grid_columnconfigure((0, 1), weight=1)

        ai_frame = ctk.CTkFrame(input_frame, fg_color="#1C1C1E", corner_radius=10)
        ai_frame.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        ctk.CTkLabel(ai_frame, text="🤖 Yapay Zeka Çıktısı (Hypothesis)", font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"), text_color="#4DA8DA").pack(anchor="w", padx=15, pady=(10, 0))
        self.ai_textbox = ctk.CTkTextbox(ai_frame, height=130, font=self.main_font, fg_color="#2C2C2E", text_color="white", border_width=1, border_color="#3A3A3C")
        self.ai_textbox.pack(fill="both", expand=True, padx=15, pady=15)
        self.ai_textbox.insert("0.0", "Yapay zeka modelleri günümüzde çok hızlı gelişiyor. Öğrenme kapasiteleri arttıkça hata payları düşüyor.")

        ref_frame = ctk.CTkFrame(input_frame, fg_color="#1C1C1E", corner_radius=10)
        ref_frame.grid(row=0, column=1, padx=(10, 0), sticky="nsew")
        ctk.CTkLabel(ref_frame, text="📝 Referans Metin (Premise)", font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"), text_color="#A8E6CF").pack(anchor="w", padx=15, pady=(10, 0))
        self.ref_textbox = ctk.CTkTextbox(ref_frame, height=130, font=self.main_font, fg_color="#2C2C2E", text_color="white", border_width=1, border_color="#3A3A3C")
        self.ref_textbox.pack(fill="both", expand=True, padx=15, pady=15)
        self.ref_textbox.insert("0.0", "Günümüzde yapay zeka sistemleri oldukça süratli bir gelişim gösteriyor. Modellerin kavrama yeteneği yükseldikçe yanılma oranları azalıyor.")

        # 3. Butonlar
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.grid(row=2, column=0, columnspan=2, pady=5)

        ctk.CTkButton(self.btn_frame, text="🚀 Genişletilmiş NLP Analizi", font=ctk.CTkFont(weight="bold", size=15), command=self.start_nlp_thread, width=220, height=45, fg_color="#007AFF", hover_color="#005BB5").pack(side="left", padx=10)
        ctk.CTkButton(self.btn_frame, text="📊 Okunabilirlik Analizi", font=ctk.CTkFont(weight="bold", size=15), command=self.start_readability_thread, width=220, height=45, fg_color="#34C759", hover_color="#248A3D").pack(side="left", padx=10)
        ctk.CTkButton(self.btn_frame, text="🗑️ Temizle", font=ctk.CTkFont(weight="bold", size=15), command=self.clear_results, fg_color="#FF3B30", hover_color="#C72C24", width=220, height=45).pack(side="left", padx=10)

        # 4. Sonuçların Gösterileceği Kaydırılabilir Alan
        self.scrollable_frame = ctk.CTkScrollableFrame(self, fg_color="#121212")
        self.scrollable_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=15, sticky="nsew")
        self.grid_rowconfigure(3, weight=10)

        self.log_console = ctk.CTkTextbox(self.scrollable_frame, height=120, font=self.console_font, fg_color="#000000", text_color="#00FF00", border_width=1, border_color="#333333")
        self.log_console.pack(fill="x", pady=(0, 15))
        self.log_console.insert("0.0", "[SİSTEM]: Hazır. Yüksek Lisans tezi analizleri bekleniyor...\n")
        self.log_console.configure(state="disabled")
        
        self.results_container = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        self.results_container.pack(fill="both", expand=True)

    def write_log(self, text):
        """ Log ekranına (Thread içinden) güvenli metin yazdırma metodu """
        self.log_console.configure(state="normal")
        self.log_console.insert("end", text + "\n")
        self.log_console.see("end")
        self.log_console.configure(state="disabled")
        self.update_idletasks() 

    def safe_log(self, text):
        """ Thread'lerin (Arka plan iş parçacıklarının) arayüzü kilitlememesi için logu ana thread'e paslar """
        self.after(0, self.write_log, text)

    def clear_results(self):
        for widget in self.results_container.winfo_children():
            widget.destroy()
        self.log_console.configure(state="normal")
        self.log_console.delete("0.0", "end")
        self.log_console.insert("0.0", "[SİSTEM]: Ekran temizlendi. Yeni analiz bekleniyor...\n")
        self.log_console.configure(state="disabled")

    # ==========================================
    # ASENKRON İŞLEM YÖNETİMİ (THREADING)
    # Arayüzün donmaması için işlemleri arkada yürütür.
    # ==========================================
    def start_nlp_thread(self):
        for widget in self.results_container.winfo_children(): widget.destroy()
        
        ai_text = self.ai_textbox.get("0.0", "end").strip()
        ref_text = self.ref_textbox.get("0.0", "end").strip()

        if not ai_text or not ref_text:
            self.write_log("⚠️ HATA: Lütfen iki metni de girin.")
            return

        self.write_log("-" * 60)
        self.write_log("🚀 Genişletilmiş NLP Analizi Başlıyor...")
        threading.Thread(target=self._run_nlp, args=(ai_text, ref_text), daemon=True).start()

    def _run_nlp(self, ai_text, ref_text):
        result_data = process_nlp(ai_text, ref_text, self.safe_log)
        self.after(0, lambda: build_nlp_dashboard(self.results_container, result_data, self.main_font, self.console_font))

    def start_readability_thread(self):
        for widget in self.results_container.winfo_children(): widget.destroy()
        
        ai_text = self.ai_textbox.get("0.0", "end").strip()
        ref_text = self.ref_textbox.get("0.0", "end").strip()

        if not ai_text or not ref_text:
            self.write_log("⚠️ HATA: Lütfen iki metni de girin.")
            return

        self.write_log("-" * 60)
        threading.Thread(target=self._run_readability, args=(ai_text, ref_text), daemon=True).start()

    def _run_readability(self, ai_text, ref_text):
        ai_res, ref_res = process_readability(ai_text, ref_text, self.safe_log)
        self.after(0, lambda: build_readability_gui(self.results_container, ai_res, ref_res, self.main_font, self.console_font))

if __name__ == "__main__":
    app = TextAnalyzerApp()
    app.mainloop()
