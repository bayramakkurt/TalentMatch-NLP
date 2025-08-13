from pymongo import MongoClient
from gridfs import GridFS
from typing import Dict, List, Optional
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv()

def fix_mongo_ids(doc):
    """MongoDB ObjectId'lerini string'e çevir"""
    if isinstance(doc, list):
        return [fix_mongo_ids(d) for d in doc]
    if isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                new_doc[k] = str(v)
            elif isinstance(v, dict) or isinstance(v, list):
                new_doc[k] = fix_mongo_ids(v)
            else:
                new_doc[k] = v
        return new_doc
    return doc

class Database:
    def __init__(self):
        """MongoDB bağlantısını ve GridFS'i başlat"""
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
        self.db = self.client.talentmatch
        self.fs = GridFS(self.db)
        
        # Koleksiyonlar
        self.cv_metadata = self.db.cv_metadata  # Tek koleksiyon kullan
        self.job_postings = self.db.job_postings
        self.matches = self.db.matches
        
    def save_cv_file(self, file_content: bytes, filename: str, content_type: str):
        """Dosyayı GridFS'e kaydeder ve file_id döner"""
        file_id = self.fs.put(
            file_content, 
            filename=filename, 
            content_type=content_type,
            upload_date=datetime.utcnow()
        )
        return file_id

    def save_cv_metadata(self, file_id, cv_info, filename: str):
        """CV bilgilerini metadata koleksiyonuna kaydeder"""
        metadata = {
            "file_id": file_id,
            "filename": filename,
            "upload_date": datetime.utcnow(),
            "cv_data": {
                "names": cv_info.names,
                "education": cv_info.education,
                "experience": cv_info.experience,
                "skills": cv_info.skills,
                "contact_info": cv_info.contact_info,
                "summary": cv_info.summary
            }
        }
        result = self.cv_metadata.insert_one(metadata)
        return str(result.inserted_id)
    
    def get_cv(self, cv_id: str) -> Optional[Dict]:
        """CV meta verilerini al"""
        try:
            cv_data = self.cv_metadata.find_one({"_id": ObjectId(cv_id)})
            return fix_mongo_ids(cv_data) if cv_data else None
        except Exception as e:
            print(f"CV getirme hatası: {e}")
            return None
    
    def get_cv_file(self, file_id) -> Optional[bytes]:
        """GridFS'den dosya içeriğini al"""
        try:
            if isinstance(file_id, str):
                file_id = ObjectId(file_id)
            file_data = self.fs.get(file_id)
            return file_data.read()
        except Exception as e:
            print(f"Dosya getirme hatası: {e}")
            return None
    
    def store_job_posting(self, job_data: Dict) -> str:
        """İş ilanını veritabanına kaydet"""
        job_data["created_at"] = datetime.utcnow()
        job_data["status"] = "active"
        result = self.job_postings.insert_one(job_data)
        return str(result.inserted_id)
    
    def get_job_posting(self, job_id: str) -> Optional[Dict]:
        """İş ilanını veritabanından al"""
        try:
            doc = self.job_postings.find_one({"_id": ObjectId(job_id)})
            return fix_mongo_ids(doc) if doc else None
        except Exception as e:
            print(f"İş ilanı getirme hatası: {e}")
            return None
    
    def store_match(self, job_id: str, candidate_id: str, match_data: Dict) -> str:
        """Eşleşme sonucunu veritabanına kaydet"""
        match_record = {
            "job_id": job_id,
            "candidate_id": candidate_id,
            "match_percentage": match_data.get("match_percentage", 0),
            "missing_skills": match_data.get("missing_skills", []),
            "explanation": match_data.get("explanation", ""),
            "notification_sent": False,
            "created_at": datetime.utcnow()
        }
        result = self.matches.insert_one(match_record)
        return str(result.inserted_id)
    
    def get_matches_for_job(self, job_id: str) -> List[Dict]:
        """Bir iş ilanı için tüm eşleşmeleri al"""
        try:
            matches = list(self.matches.find({"job_id": job_id}).sort("match_percentage", -1))
            return fix_mongo_ids(matches)
        except Exception as e:
            print(f"Eşleşmeler getirme hatası: {e}")
            return []
    
    def get_all_candidates(self) -> List[Dict]:
        """Tüm adayları al (CV metadata'sından)"""
        try:
            candidates = list(self.cv_metadata.find())
            return fix_mongo_ids(candidates)
        except Exception as e:
            print(f"Adaylar getirme hatası: {e}")
            return []
    
    def get_all_job_postings(self) -> List[Dict]:
        """Tüm iş ilanlarını al"""
        try:
            jobs = list(self.job_postings.find().sort("created_at", -1))
            return fix_mongo_ids(jobs)
        except Exception as e:
            print(f"İş ilanları getirme hatası: {e}")
            return []
    
    def update_match_parameters(self, job_id: str, parameters: Dict) -> bool:
        """İş ilanı için eşleştirme parametrelerini güncelle"""
        try:
            result = self.job_postings.update_one(
                {"_id": ObjectId(job_id)},
                {"$set": {"matching_parameters": parameters, "updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Parametreler güncelleme hatası: {e}")
            return False
    
    def mark_notification_sent(self, match_id: str) -> bool:
        """Bildirim gönderildi olarak işaretle"""
        try:
            result = self.matches.update_one(
                {"_id": ObjectId(match_id)},
                {"$set": {"notification_sent": True, "notification_sent_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Bildirim güncelleme hatası: {e}")
            return False
    
    def get_unsent_matches(self) -> List[Dict]:
        """Henüz bildirim gönderilmemiş eşleşmeleri al"""
        try:
            matches = list(self.matches.find({"notification_sent": {"$ne": True}}))
            return fix_mongo_ids(matches)
        except Exception as e:
            print(f"Gönderilmemiş eşleşmeler getirme hatası: {e}")
            return []