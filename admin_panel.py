"""
Basit Admin Paneli - Streamlit ile
Admin işlemleri için web arayüzü
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import os
from typing import List, Dict

# API Base URL
API_BASE_URL = "http://host.docker.internal:8000"

def get_api_url(endpoint: str) -> str:
    return f"{API_BASE_URL}{endpoint}"

def make_api_request(method: str, endpoint: str, data=None, files=None):
    """API isteği gönder"""
    url = get_api_url(endpoint)
    
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url)
        
        return response.json() if response.headers.get('content-type') == 'application/json' else response.text
    except Exception as e:
        st.error(f"API isteği başarısız: {e}")
        return None

def main():
    st.set_page_config(
        page_title="TalentMatch NLP Admin",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 TalentMatch NLP Admin Paneli")
    st.sidebar.title("Admin İşlemleri")
    
    # Sidebar menü
    menu = st.sidebar.selectbox(
        "İşlem Seçin",
        [
            "Ana Sayfa",
            "CV Yönetimi", 
            "İş İlanları",
            "Eşleştirmeler",
            "Bildirim Gönder",
            "İstatistikler"
        ]
    )
    
    if menu == "Ana Sayfa":
        show_dashboard()
    elif menu == "CV Yönetimi":
        show_cv_management()
    elif menu == "İş İlanları":
        show_job_management()
    elif menu == "Eşleştirmeler":
        show_matches()
    elif menu == "Bildirim Gönder":
        show_notification_panel()
    elif menu == "İstatistikler":
        show_statistics()

def show_dashboard():
    """Ana sayfa dashboard"""
    st.header("📊 Sistem Özeti")
    
    # İstatistikleri al
    stats = make_api_request("GET", "/statistics")
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Aday", stats.get("candidates", 0))
        
        with col2:
            st.metric("Aktif İş İlanları", stats.get("job_postings", 0))
        
        with col3:
            st.metric("Toplam Eşleşme", stats.get("total_matches", 0))
        
        with col4:
            st.metric("Bekleyen Bildirimler", stats.get("unsent_notifications", 0))
    
    # Son aktiviteler
    st.subheader("🔄 Sistem Durumu")
    
    # Health check
    health = make_api_request("GET", "/health")
    if health:
        if health.get("status") == "healthy":
            st.success("✅ Sistem çalışıyor")
        else:
            st.error("❌ Sistem hatası")
        
        st.json(health)

def show_cv_management():
    """CV yönetimi sayfası"""
    st.header("📄 CV Yönetimi")
    
    tab1, tab2 = st.tabs(["CV Listesi", "CV Yükle"])
    
    with tab1:
        st.subheader("Kayıtlı CV'ler")
        
        # CV'leri listele
        candidates = make_api_request("GET", "/candidates")
        
        if candidates and candidates.get("candidates"):
            cv_list = candidates["candidates"]
            
            # DataFrame oluştur
            df_data = []
            for cv in cv_list:
                cv_data = cv.get("cv_data", {})
                names = cv_data.get("names", ["Bilinmiyor"])
                email = cv_data.get("contact_info", {}).get("email", "Belirtilmemiş")
                skills_count = len(cv_data.get("skills", []))
                
                df_data.append({
                    "ID": cv.get("_id", ""),
                    "Ad Soyad": names[0] if names else "Bilinmiyor",
                    "E-posta": email,
                    "Beceri Sayısı": skills_count,
                    "Dosya": cv.get("filename", "")
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # CV detaylarını göster
            selected_id = st.selectbox("Detay görmek için CV seçin:", options=[""] + [cv.get("_id", "") for cv in cv_list])
            
            if selected_id:
                selected_cv = next((cv for cv in cv_list if cv.get("_id") == selected_id), None)
                if selected_cv:
                    st.subheader("CV Detayları")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**İsimler:**", selected_cv.get("cv_data", {}).get("names", []))
                        st.write("**İletişim:**", selected_cv.get("cv_data", {}).get("contact_info", {}))
                    
                    with col2:
                        st.write("**Beceriler:**", selected_cv.get("cv_data", {}).get("skills", []))
                    
                    st.write("**Özet:**")
                    st.text_area("", value=selected_cv.get("cv_data", {}).get("summary", ""), height=150, disabled=True)
                    
                    # Silme butonu
                    if st.button("🗑️ Bu CV'yi Sil", type="secondary"):
                        result = make_api_request("DELETE", f"/candidates/{selected_id}")
                        if result:
                            st.success("CV başarıyla silindi!")
                            st.rerun()
        else:
            st.info("Henüz hiç CV kaydedilmemiş.")
    
    with tab2:
        st.subheader("Yeni CV Yükle")
        
        uploaded_file = st.file_uploader(
            "CV Dosyasını Seçin", 
            type=['pdf', 'docx'],
            help="PDF veya DOCX formatında CV yükleyebilirsiniz"
        )
        
        if uploaded_file and st.button("CV'yi İşle"):
            with st.spinner("CV işleniyor..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                result = make_api_request("POST", "/upload-cv", files=files)
                
                if result and result.get("status") == "success":
                    st.success("CV başarıyla yüklendi ve işlendi!")
                    st.json(result)
                else:
                    st.error(f"CV işleme hatası: {result}")

def show_job_management():
    """İş ilanları yönetimi"""
    st.header("💼 İş İlanları")
    
    tab1, tab2 = st.tabs(["İlan Listesi", "Yeni İlan"])
    
    with tab1:
        st.subheader("Aktif İş İlanları")
        
        jobs = make_api_request("GET", "/job-postings")
        
        if jobs and jobs.get("job_postings"):
            job_list = jobs["job_postings"]
            
            for job in job_list:
                with st.expander(f"🏢 {job.get('company', 'Bilinmeyen')} - {job.get('title', 'Başlıksız')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Şirket:**", job.get("company", ""))
                        st.write("**Pozisyon:**", job.get("title", ""))
                        st.write("**Lokasyon:**", job.get("location", ""))
                    
                    with col2:
                        st.write("**Oluşturulma:**", job.get("created_at", ""))
                        matches = make_api_request("GET", f"/job-postings/{job.get('_id')}/matches")
                        match_count = len(matches.get("matches", [])) if matches else 0
                        st.write("**Eşleşme Sayısı:**", match_count)
                    
                    st.write("**Açıklama:**")
                    st.write(job.get("description", ""))
                    
                    st.write("**Gereksinimler:**")
                    for req in job.get("requirements", []):
                        st.write(f"• {req}")
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        if st.button(f"🔍 Eşleştirme Yap", key=f"match_{job.get('_id')}"):
                            with st.spinner("Eşleştirme yapılıyor..."):
                                result = make_api_request("POST", f"/match-candidates/{job.get('_id')}")
                                if result:
                                    st.success(f"Eşleştirme tamamlandı! {result.get('count', 0)} aday bulundu.")
                    
                    with col4:
                        if st.button(f"🗑️ İlanı Sil", key=f"delete_{job.get('_id')}", type="secondary"):
                            result = make_api_request("DELETE", f"/job-postings/{job.get('_id')}")
                            if result:
                                st.success("İlan silindi!")
                                st.rerun()
        else:
            st.info("Henüz hiç iş ilanı eklenmemiş.")
    
    with tab2:
        st.subheader("Yeni İş İlanı Ekle")
        
        with st.form("new_job_form"):
            company = st.text_input("Şirket Adı*")
            title = st.text_input("Pozisyon*")
            location = st.text_input("Lokasyon*")
            description = st.text_area("İş Açıklaması*", height=150)
            
            st.write("**Gereksinimler** (her satıra bir gereksinim)")
            requirements_text = st.text_area("", height=100, placeholder="Python programlama\nDjango framework\n3+ yıl deneyim")
            
            if st.form_submit_button("İlanı Kaydet"):
                requirements = [req.strip() for req in requirements_text.split('\n') if req.strip()]
                
                if company and title and location and description and requirements:
                    job_data = {
                        "company": company,
                        "title": title,
                        "location": location,
                        "description": description,
                        "requirements": requirements
                    }
                    
                    result = make_api_request("POST", "/job-posting", job_data)
                    
                    if result:
                        st.success("İş ilanı başarıyla oluşturuldu!")
                        st.balloons()
                else:
                    st.error("Tüm alanları doldurun!")

def show_matches():
    """Eşleştirmeler sayfası"""
    st.header("🎯 Eşleştirme Sonuçları")
    
    # İş ilanlarını al
    jobs = make_api_request("GET", "/job-postings")
    
    if jobs and jobs.get("job_postings"):
        job_options = {f"{job.get('company')} - {job.get('title')}": job.get('_id') 
                      for job in jobs["job_postings"]}
        
        selected_job_display = st.selectbox("İş İlanı Seçin:", options=list(job_options.keys()))
        
        if selected_job_display:
            job_id = job_options[selected_job_display]
            
            # Eşleşmeleri al
            matches = make_api_request("GET", f"/job-postings/{job_id}/matches")
            
            if matches and matches.get("matches"):
                st.subheader(f"📊 {selected_job_display} - Eşleşmeler")
                
                match_list = matches["matches"]
                
                # Eşleşmeleri göster
                for i, match in enumerate(match_list):
                    # Aday bilgilerini al
                    candidate = make_api_request("GET", f"/candidates/{match['candidate_id']}")
                    
                    if candidate:
                        cv_data = candidate.get("cv_data", {})
                        names = cv_data.get("names", ["Bilinmeyen"])
                        email = cv_data.get("contact_info", {}).get("email", "Belirtilmemiş")
                        
                        # Eşleşme kartı
                        with st.container():
                            col1, col2, col3 = st.columns([2, 3, 1])
                            
                            with col1:
                                st.write(f"**👤 {(cv_data.get('names') or ['Bilinmeyen'])[0]}**")
                                st.write(f"📧 {email}")
                                
                                # Eşleşme yüzdesi
                                percentage = match.get("match_percentage", 0)
                                if percentage >= 70:
                                    st.success(f"🎯 %{percentage:.1f} Eşleşme")
                                elif percentage >= 50:
                                    st.warning(f"⚠️ %{percentage:.1f} Eşleşme")
                                else:
                                    st.error(f"❌ %{percentage:.1f} Eşleşme")
                            
                            with col2:
                                st.write("**Açıklama:**", match.get("explanation", ""))
                                
                                missing_skills = match.get("missing_skills", [])
                                if missing_skills:
                                    st.write("**Eksik Beceriler:**", ", ".join(missing_skills[:3]))
                            
                            with col3:
                                # Bildirim durumu
                                if match.get("notification_sent"):
                                    st.success("✅ Gönderildi")
                                else:
                                    st.info("⏳ Bekliyor")
                            
                            st.divider()
            else:
                st.info("Bu iş ilanı için henüz eşleştirme yapılmamış.")
    else:
        st.info("Henüz hiç iş ilanı eklenmemiş.")

def show_notification_panel():
    """Bildirim gönderme paneli"""
    st.header("📧 Bildirim Gönderme")
    
    tab1, tab2 = st.tabs(["Bekleyen Bildirimler", "Manuel Gönderim"])
    
    with tab1:
        st.subheader("⏳ Bekleyen Bildirimler")
        
        # Gönderilmemiş bildirimleri al
        unsent = make_api_request("GET", "/unsent-notifications")
        
        if unsent and unsent.get("unsent_notifications"):
            notifications = unsent["unsent_notifications"]
            
            st.write(f"**Toplam bekleyen bildirim: {len(notifications)}**")
            
            # Tablo halinde göster
            df_data = []
            for notif in notifications:
                df_data.append({
                    "İş İlanı": f"{notif.get('company', '')} - {notif.get('job_title', '')}",
                    "Aday": notif.get('candidate_name', 'Bilinmiyor'),
                    "E-posta": notif.get('candidate_email', ''),
                    "Eşleşme": f"%{notif.get('match_percentage', 0):.1f}",
                    "Tarih": notif.get('created_at', '')[:10] if notif.get('created_at') else ''
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Toplu gönderme butonu
            if st.button("📤 Tüm Bildirimleri Gönder", type="primary"):
                with st.spinner("Bildirimler gönderiliyor..."):
                    # Her iş ilanı için ayrı ayrı gönder
                    job_groups = {}
                    for notif in notifications:
                        job_title = notif.get('job_title', '')
                        if job_title not in job_groups:
                            job_groups[job_title] = []
                        job_groups[job_title].append(notif)
                    
                    total_sent = 0
                    for job_title, job_notifications in job_groups.items():
                        # İlk bildirimden job_id'yi al (bu kısım API'ya göre ayarlanmalı)
                        st.write(f"📧 {job_title} için {len(job_notifications)} bildirim gönderiliyor...")
                        total_sent += len(job_notifications)
                    
                    if total_sent > 0:
                        st.success(f"✅ {total_sent} bildirim gönderildi!")
                        st.balloons()
                        # Sayfayı yenile
                        st.rerun()
        else:
            st.info("Bekleyen bildirim bulunmuyor.")
    
    with tab2:
        st.subheader("📤 Manuel Bildirim Gönderimi")
        
        # İş ilanlarını listele
        jobs = make_api_request("GET", "/job-postings")
        
        if jobs and jobs.get("job_postings"):
            job_options = {f"{job.get('company')} - {job.get('title')}": job.get('_id') 
                          for job in jobs["job_postings"]}
            
            selected_job = st.selectbox("İş İlanı Seçin:", options=list(job_options.keys()))
            
            if selected_job:
                job_id = job_options[selected_job]
                
                # Eşleşmeleri al
                matches = make_api_request("GET", f"/job-postings/{job_id}/matches")
                
                if matches and matches.get("matches"):
                    st.write(f"**{len(matches['matches'])} eşleşme bulundu**")
                    
                    # Gönderilecek adayları seç
                    candidate_options = {}
                    for match in matches["matches"]:
                        candidate = make_api_request("GET", f"/candidates/{match['candidate_id']}")
                        if candidate:
                            cv_data = candidate.get("cv_data", {})
                            names = cv_data.get("names", ["Bilinmiyor"])
                            email = cv_data.get("contact_info", {}).get("email", "")
                            
                            if email:  # Sadece e-postası olanları göster
                                key = f"{(cv_data.get('names') or ['Bilinmeyen'])[0]} ({email}) - %{match.get('match_percentage', 0):.1f}"
                                candidate_options[key] = match['candidate_id']
                    
                    if candidate_options:
                        selected_candidates = st.multiselect(
                            "Bildirim gönderilecek adayları seçin:",
                            options=list(candidate_options.keys())
                        )
                        
                        if selected_candidates and st.button("🚀 Seçili Adaylara Gönder"):
                            candidate_ids = [candidate_options[name] for name in selected_candidates]
                            
                            # Bildirim gönder
                            notification_data = {
                                "job_id": job_id,
                                "candidate_ids": candidate_ids
                            }
                            
                            with st.spinner("Bildirimler gönderiliyor..."):
                                result = make_api_request("POST", "/send-notifications", notification_data)
                                
                                if result:
                                    st.success(f"✅ {len(selected_candidates)} adaya bildirim gönderildi!")
                                    st.json(result)
                                else:
                                    st.error("Bildirim gönderiminde hata oluştu!")
                    else:
                        st.warning("E-posta adresi olan aday bulunamadı.")
                else:
                    st.info("Bu iş ilanı için eşleşme bulunamadı.")
        else:
            st.info("Henüz iş ilanı bulunmuyor.")

def show_statistics():
    """İstatistikler sayfası"""
    st.header("📊 Sistem İstatistikleri")
    
    # Ana istatistikler
    stats = make_api_request("GET", "/statistics")
    
    if stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Genel Bilgiler")
            st.metric("Toplam Aday", stats.get("candidates", 0))
            st.metric("Aktif İş İlanları", stats.get("job_postings", 0))
            st.metric("Toplam Eşleşme", stats.get("total_matches", 0))
            st.metric("Bekleyen Bildirimler", stats.get("unsent_notifications", 0))
        
        with col2:
            st.subheader("🎯 Eşleşme Analizi")
            
            # İş ilanları ve eşleşmeleri
            jobs = make_api_request("GET", "/job-postings")
            if jobs and jobs.get("job_postings"):
                job_stats = []
                
                for job in jobs["job_postings"]:
                    matches = make_api_request("GET", f"/job-postings/{job.get('_id')}/matches")
                    match_count = len(matches.get("matches", [])) if matches else 0
                    
                    job_stats.append({
                        "İş İlanı": f"{job.get('company', '')} - {job.get('title', '')}",
                        "Eşleşme Sayısı": match_count
                    })
                
                if job_stats:
                    df = pd.DataFrame(job_stats)
                    st.dataframe(df, use_container_width=True)
                    
                    # Grafik
                    if len(df) > 0:
                        st.bar_chart(df.set_index("İş İlanı")["Eşleşme Sayısı"])
    
    # Sistem durumu
    st.subheader("🖥️ Sistem Durumu")
    health = make_api_request("GET", "/health")
    
    if health:
        if health.get("status") == "healthy":
            st.success("✅ Sistem sağlıklı çalışıyor")
        else:
            st.error("❌ Sistem hatası tespit edildi")
        
        st.json(health)
    
    # Log bilgileri (basit)
    st.subheader("📝 Son Aktiviteler")
    st.info("Log görüntüleme özelliği yakında eklenecek...")

if __name__ == "__main__":
    main()