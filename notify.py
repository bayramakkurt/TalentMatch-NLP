import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
from jinja2 import Template
from typing import Dict, Optional
import logging

load_dotenv()

class NotificationService:
    def __init__(self):
        # SMTP ayarları
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        
        # Email template'i
        self.email_template = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>İş Fırsatı Eşleşmesi</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #4CAF50; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f9f9f9; }
        .job-info { background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .match-info { background-color: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .skills-missing { color: #e74c3c; }
        .match-percentage { font-size: 24px; font-weight: bold; color: #2ecc71; }
        .footer { text-align: center; padding: 20px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Yeni İş Fırsatı Eşleşmesi!</h1>
        </div>
        
        <div class="content">
            <p>Merhaba {% if candidate_name %}{{ candidate_name }}{% else %}Değerli Aday{% endif %},</p>
            
            <p>CV'niz aşağıdaki iş ilanı ile <strong class="match-percentage">%{{ match_percentage }}</strong> oranında eşleşti!</p>
            
            <div class="job-info">
                <h2>📋 İş İlanı Detayları</h2>
                <p><strong>Pozisyon:</strong> {{ job_title }}</p>
                <p><strong>Şirket:</strong> {{ company_name }}</p>
                <p><strong>Lokasyon:</strong> {{ location }}</p>
                <p><strong>İş Tanımı:</strong></p>
                <p>{{ job_description }}</p>
                
                <h3>📋 Aranan Nitelikler:</h3>
                <ul>
                {% for requirement in requirements %}
                    <li>{{ requirement }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <div class="match-info">
                <h3>📊 Eşleşme Analizi</h3>
                <p>{{ explanation }}</p>
                
                {% if missing_skills %}
                <p class="skills-missing"><strong>Eksik Beceriler:</strong> {{ missing_skills|join(', ') }}</p>
                <p><small>Bu becerileri geliştirerek eşleşme oranınızı artırabilirsiniz.</small></p>
                {% endif %}
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <p><strong>İlgileniyor musunuz?</strong></p>
                <p>Bu fırsat hakkında daha fazla bilgi almak için HR ekibimizle iletişime geçebilirsiniz.</p>
            </div>
        </div>
        
        <div class="footer">
            <p>Bu e-posta TalentMatch NLP sistemi tarafından otomatik olarak oluşturulmuştur.</p>
            <p>© 2024 TalentMatch - Tüm hakları saklıdır.</p>
        </div>
    </div>
</body>
</html>
        """

    def send_email(self, to_email: str, subject: str, html_body: str) -> bool:
        """E-posta gönder"""
        try:
            msg = MIMEMultipart('alternative')
            msg["From"] = self.smtp_username
            msg["To"] = to_email
            msg["Subject"] = subject

            html_part = MIMEText(html_body, "html", "utf-8")
            msg.attach(html_part)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            print(f"E-posta başarıyla gönderildi: {to_email}")
            return True
            
        except Exception as e:
            print(f"E-posta gönderilirken hata oluştu: {e}")
            return False

    def send_match_notification(self, candidate_email: str, job_data: Dict, match_data: Dict, candidate_data: Dict = None) -> bool:
        """
        İş eşleşmesi bildirimi gönder
        
        Args:
            candidate_email: Adayın e-posta adresi
            job_data: İş ilanı verileri
            match_data: Eşleşme verileri (match_percentage, missing_skills, explanation)
            candidate_data: Aday verileri (opsiyonel)
        """
        try:
            # Template'i render et
            template = Template(self.email_template)
            
            # Aday ismini bul
            candidate_name = None
            if candidate_data and candidate_data.get("cv_data", {}).get("names"):
                names = candidate_data["cv_data"]["names"]
                if names:
                    candidate_name = names[0]
            
            html_body = template.render(
                candidate_name=candidate_name,
                match_percentage=round(match_data.get("match_percentage", 0), 1),
                job_title=job_data.get("title", "Belirtilmemiş"),
                company_name=job_data.get("company", "Belirtilmemiş"),
                location=job_data.get("location", "Belirtilmemiş"),
                job_description=job_data.get("description", "Açıklama bulunmuyor"),
                requirements=job_data.get("requirements", []),
                explanation=match_data.get("explanation", ""),
                missing_skills=match_data.get("missing_skills", [])
            )
            
            subject = f"🎯 Yeni İş Fırsatı - %{round(match_data.get('match_percentage', 0), 1)} Eşleşme"
            
            return self.send_email(candidate_email, subject, html_body)
            
        except Exception as e:
            print(f"Eşleşme bildirimi gönderilirken hata: {e}")
            return False

    def send_bulk_notifications(self, notifications: list) -> Dict[str, int]:
        """
        Toplu bildirim gönder
        
        Args:
            notifications: Liste şeklinde bildirim verileri
            Her eleman: {
                'candidate_email': str,
                'job_data': dict,
                'match_data': dict,
                'candidate_data': dict
            }
        
        Returns:
            {
                'sent': int,        # Başarıyla gönderildi
                'failed': int,      # Başarısız
                'total': int        # Toplam
            }
        """
        results = {'sent': 0, 'failed': 0, 'total': len(notifications)}
        
        for notification in notifications:
            try:
                success = self.send_match_notification(
                    notification['candidate_email'],
                    notification['job_data'],
                    notification['match_data'],
                    notification.get('candidate_data')
                )
                
                if success:
                    results['sent'] += 1
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                print(f"Bildirim gönderim hatası: {e}")
                results['failed'] += 1
        
        return results

    def validate_email_config(self) -> bool:
        """E-posta konfigürasyonunu doğrula"""
        required_configs = [
            self.smtp_username,
            self.smtp_password,
            self.smtp_server
        ]
        
        if not all(required_configs):
            print("SMTP konfigürasyonu eksik. .env dosyasını kontrol edin.")
            return False
            
        return True