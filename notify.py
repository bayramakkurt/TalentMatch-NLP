import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
from jinja2 import Template
from pymongo import MongoClient
from bson import ObjectId

load_dotenv()

class NotificationService:
    def __init__(self):
        # SMTP ayarları
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        
        # MongoDB ayarları
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        mongo_db = os.getenv("MONGO_DB", "cv_database")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[mongo_db]
        self.metadata_collection = self.db["cv_metadata"]

    def send_email(self, to_email: str, subject: str, html_body: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg["From"] = self.smtp_username
            msg["To"] = to_email
            msg["Subject"] = subject

            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            return True
        except Exception as e:
            print(f"E-posta gönderilirken hata oluştu: {e}")
            return False

    def render_email_body(self, cv_data: dict) -> str:
        with open("cv_email_template.html", "r", encoding="utf-8") as file:
            template_str = file.read()
        template = Template(template_str)
        return template.render(
            file_id=cv_data["file_id"],
            filename=cv_data["filename"],
            names=cv_data["cv_data"]["names"],
            education=cv_data["cv_data"]["education"],
            experience=cv_data["cv_data"]["experience"],
            skills=cv_data["cv_data"]["skills"],
            contact_info=cv_data["cv_data"]["contact_info"],
            summary=cv_data["cv_data"]["summary"]
        )

    def send_cv_notification(self, file_id: str) -> bool:
        """
        Verilen file_id ile veritabanından CV verisini çekip, içindeki email'e e-posta gönderir.
        """
        doc = self.metadata_collection.find_one({"file_id": ObjectId(file_id)})
        if not doc:
            print("Veri bulunamadı.")
            return False

        email = doc.get("cv_data", {}).get("contact_info", {}).get("email")
        if not email:
            print("CV içeriğinde e-posta adresi bulunamadı.")
            return False

        doc["file_id"] = str(file_id)  # string'e çevir
        html_body = self.render_email_body(doc)

        return self.send_email(email, "CV’niz başarıyla işlendi", html_body)
