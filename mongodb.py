
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
        """
        MongoDB bağlantısını ve GridFS'i başlat
        """
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
        self.db = self.client.talentmatch
        self.fs = GridFS(self.db)
        
        # Koleksiyonlar
        self.candidates = self.db.candidates
        self.job_postings = self.db.job_postings
        self.matches = self.db.matches
        self.metadata_collection = self.db.cv_metadata
        
    def save_cv_file(self, file_content: bytes, filename: str, content_type: str):
        """
        Dosyayı GridFS'e kaydeder ve file_id döner.
        """
        file_id = self.fs.put(file_content, filename=filename, content_type=content_type)
        return file_id

    def save_cv_metadata(self, file_id, cv_info, filename: str):
        """
        CV bilgilerini metadata koleksiyonuna kaydeder ve metadata_id döner.
        """
        metadata = {
            "file_id": file_id,
            "filename": filename,
            "cv_data": {
                "names": cv_info.names,
                "education": cv_info.education,
                "experience": cv_info.experience,
                "skills": cv_info.skills,
                "contact_info": cv_info.contact_info,
                "summary": cv_info.summary
            }
        }
        metadata_id = self.metadata_collection.insert_one(metadata).inserted_id
        return metadata_id
    
    def get_cv(self, cv_id: str) -> Optional[Dict]:
        """
        CV meta verilerini ve dosya içeriğini al
        """
        cv_data = self.candidates.find_one({"_id": ObjectId(cv_id)})
        if cv_data and "file_id" in cv_data:
            file_data = self.fs.get(cv_data["file_id"])
            cv_data["file_content"] = file_data.read()
            return fix_mongo_ids(cv_data)
        return None
    
    def store_job_posting(self, job_data: Dict) -> str:
        """
        İş ilanını veritabanına kaydet
        """
        job_data["created_at"] = datetime.utcnow()
        result = self.job_postings.insert_one(job_data)
        return str(result.inserted_id)
    
    def get_job_posting(self, job_id: str) -> Optional[Dict]:
        """
        İş ilanını veritabanından al
        """
        doc = self.job_postings.find_one({"_id": ObjectId(job_id)})
        return fix_mongo_ids(doc) if doc else None
    
    def store_match(self, job_id: str, candidate_id: str, match_data: Dict) -> str:
        """
        Eşleşme sonucunu veritabanına kaydet
        """
        match_data.update({
            "job_id": job_id,
            "candidate_id": candidate_id,
            "created_at": datetime.utcnow()
        })
        result = self.matches.insert_one(match_data)
        return str(result.inserted_id)
    
    def get_matches_for_job(self, job_id: str) -> List[Dict]:
        """
        Bir iş ilanı için tüm eşleşmeleri al
        """
        return fix_mongo_ids(list(self.matches.find({"job_id": job_id}).sort("match_percentage", -1)))
    
    def get_all_candidates(self) -> List[Dict]:
        """
        Tüm adayları al
        """
        return fix_mongo_ids(list(self.candidates.find()))
    
    def get_all_job_postings(self) -> List[Dict]:
        """
        Tüm iş ilanlarını al
        """
        return fix_mongo_ids(list(self.job_postings.find()))
    
    def update_match_parameters(self, job_id: str, parameters: Dict) -> bool:
        """
        İş ilanı için eşleştirme parametrelerini güncelle
        """
        result = self.job_postings.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"matching_parameters": parameters}}
        )
        return result.modified_count > 0 
