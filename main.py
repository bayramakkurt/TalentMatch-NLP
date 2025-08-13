from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import uvicorn
import os
from datetime import datetime

from cv_parse import EnhancedCVProcessor
from mongodb import Database, fix_mongo_ids
from notify import NotificationService
from vector import VectorMatcher

app = FastAPI(
    title="TalentMatch NLP",
    description="CV Analizi ve İş Eşleştirme Uygulaması",
    version="1.0.0"
)

# CORS middleware yapılandırması
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servisleri başlat
db = Database()
vector_matcher = VectorMatcher()
notification_service = NotificationService()

# Pydantic Modeller
class JobPosting(BaseModel):
    title: str 
    description: str 
    requirements: List[str] 
    location: str
    company: str 
    matching_parameters: Optional[Dict[str, Any]] = None  

class MatchParameters(BaseModel):
    min_match_percentage: float = 70.0  
    required_skills: List[str] = []  
    preferred_skills: List[str] = [] 

class NotificationRequest(BaseModel):
    match_ids: List[str]  # Gönderilecek eşleşme ID'leri

class BulkNotificationRequest(BaseModel):
    job_id: str
    candidate_ids: Optional[List[str]] = None  # Belirli adaylar (boşsa hepsi)

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "TalentMatch NLP API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "services": {
            "database": "connected",
            "email": notification_service.validate_email_config()
        }
    }

@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    """CV dosyası yükleme ve işleme (PDF/DOCX)"""
    
    # Dosya formatı kontrolü
    allowed_extensions = [".pdf", ".docx", ".doc"]
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail="Sadece PDF, DOC ve DOCX dosyaları kabul edilir"
        )

    # Dosya boyutu kontrolü (10MB limit)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Dosya boyutu 10MB'dan küçük olmalıdır"
        )

    try:
        # Dosya içeriğini oku
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Dosya içeriği boş")
        
        # CV'yi işle
        processor = EnhancedCVProcessor()
        result = processor.process_cv_file(file_content, file.filename)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return {
            "message": result["message"],
            "status": result["status"],
            "metadata_id": result.get("metadata_id"),
            "file_id": str(result.get("file_id")),
            "cv_summary": result.get("cv_data", {}).get("summary", "")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"CV upload hatası: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"CV işleme hatası: {str(e)}")

@app.get("/candidates")
async def get_all_candidates():
    """Tüm adayları listele"""
    try:
        candidates = db.get_all_candidates()
        return {
            "candidates": candidates,
            "count": len(candidates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: str):
    """Belirli bir adayın detaylarını al"""
    try:
        candidate = db.get_cv(candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Aday bulunamadı")
        return candidate
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/job-posting")
async def create_job_posting(job: JobPosting):
    """Yeni iş ilanı oluşturma"""
    try:
        # Zorunlu alanları kontrol et
        if not job.title.strip():
            raise HTTPException(status_code=400, detail="İş başlığı boş olamaz")
        if not job.company.strip():
            raise HTTPException(status_code=400, detail="Şirket adı boş olamaz")
        if not job.requirements:
            raise HTTPException(status_code=400, detail="En az bir gereksinim belirtilmelidir")
        
        job_id = db.store_job_posting(job.dict())
        return {
            "message": "İş ilanı başarıyla oluşturuldu",
            "job_id": job_id,
            "job_data": job.dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job-postings")
async def get_all_job_postings():
    """Tüm iş ilanlarını listele"""
    try:
        jobs = db.get_all_job_postings()
        return {
            "job_postings": jobs,
            "count": len(jobs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job-postings/{job_id}")
async def get_job_posting(job_id: str):
    """Belirli bir iş ilanının detaylarını al"""
    try:
        job = db.get_job_posting(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="İş ilanı bulunamadı")
        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match-candidates/{job_id}")
async def match_candidates(job_id: str):
    """Bir iş ilanı için uygun adayları bulma"""
    try:
        # İş ilanını al
        job = db.get_job_posting(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="İş ilanı bulunamadı")
        
        # Tüm adayları al
        candidates = db.get_all_candidates()
        if not candidates:
            return {
                "message": "Henüz hiç aday kaydedilmemiş",
                "matches": [],
                "count": 0
            }
        
        # Adayların summary'sini hazırla
        for candidate in candidates:
            cv_data = candidate.get("cv_data", {})
            summary = cv_data.get("summary", "")
            
            # Eğer summary yoksa, diğer alanlardan oluştur
            if not summary:
                skills = cv_data.get("skills", [])
                experience = cv_data.get("experience", [])
                education = cv_data.get("education", [])
                
                summary_parts = []
                if skills:
                    summary_parts.append(f"Beceriler: {', '.join(skills[:5])}")
                if experience:
                    exp_titles = [exp.get("position", "") for exp in experience if exp.get("position")]
                    if exp_titles:
                        summary_parts.append(f"Deneyim: {', '.join(exp_titles[:3])}")
                if education:
                    edu_info = [edu.get("institution", "") for edu in education if edu.get("institution")]
                    if edu_info:
                        summary_parts.append(f"Eğitim: {', '.join(edu_info[:2])}")
                
                summary = "; ".join(summary_parts) if summary_parts else "Detay bilgi bulunamadı"
            
            candidate["summary"] = summary
        
        # Vektör indeksini oluştur
        vector_matcher.create_index(candidates)
        
        # İş ilanı metnini hazırla
        job_text = f"{job['title']} {job['description']} {' '.join(job['requirements'])}"
        
        # Eşleşmeleri bul
        matches = vector_matcher.find_matches(job_text, k=min(10, len(candidates)))
        
        # Her eşleşmeyi veritabanına kaydet
        saved_matches = []
        for match in matches:
            try:
                match_id = db.store_match(job_id, match["candidate_id"], match)
                match["match_id"] = match_id
                saved_matches.append(match)
            except Exception as e:
                print(f"Eşleşme kayıt hatası: {e}")
        
        return {
            "message": f"{len(saved_matches)} aday eşleşmesi bulundu",
            "matches": saved_matches,
            "count": len(saved_matches),
            "job_info": {
                "title": job["title"],
                "company": job["company"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Eşleştirme hatası: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job-postings/{job_id}/matches")
async def get_job_matches(job_id: str):
    """Bir iş ilanı için tüm eşleşmeleri alma"""
    try:
        matches = db.get_matches_for_job(job_id)
        return {
            "matches": matches,
            "count": len(matches)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/job-postings/{job_id}/parameters")
async def update_match_parameters(job_id: str, parameters: MatchParameters):
    """İş ilanı için eşleştirme parametrelerini güncelleme"""
    try:
        success = db.update_match_parameters(job_id, parameters.dict())
        if not success:
            raise HTTPException(status_code=404, detail="İş ilanı bulunamadı")
        
        return {"message": "Eşleştirme parametreleri başarıyla güncellendi"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send-notifications")
async def send_notifications_manual(request: BulkNotificationRequest):
    """
    Belirli bir iş ilanı için eşleşen adaylara manuel olarak bildirim gönder
    Admin panelinden çağrılacak
    """
    try:
        # İş ilanını kontrol et
        job = db.get_job_posting(request.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="İş ilanı bulunamadı")
        
        # Eşleşmeleri al
        matches = db.get_matches_for_job(request.job_id)
        if not matches:
            raise HTTPException(status_code=404, detail="Bu iş ilanı için eşleşme bulunamadı")
        
        # Belirli adaylar seçildiyse filtrele
        if request.candidate_ids:
            matches = [m for m in matches if m["candidate_id"] in request.candidate_ids]
        
        # Bildirim gönderilecek eşleşmeleri hazırla
        notifications_to_send = []
        for match in matches:
            # Aday bilgilerini al
            candidate = db.get_cv(match["candidate_id"])
            if not candidate:
                continue
                
            # E-posta adresini kontrol et
            candidate_email = candidate.get("cv_data", {}).get("contact_info", {}).get("email")
            if not candidate_email:
                print(f"Aday {match['candidate_id']} için e-posta adresi bulunamadı")
                continue
            
            notifications_to_send.append({
                'candidate_email': candidate_email,
                'job_data': job,
                'match_data': match,
                'candidate_data': candidate,
                'match_id': match.get('_id')
            })
        
        if not notifications_to_send:
            return {
                "message": "Gönderilecek geçerli bildirim bulunamadı",
                "details": "Adaylarda e-posta adresi bulunmuyor olabilir"
            }
        
        # E-posta konfigürasyonunu kontrol et
        if not notification_service.validate_email_config():
            raise HTTPException(
                status_code=500, 
                detail="E-posta konfigürasyonu hatalı. SMTP ayarlarını kontrol edin."
            )
        
        # Bildirimleri gönder
        results = notification_service.send_bulk_notifications(notifications_to_send)
        
        # Başarıyla gönderilen bildirimleri işaretle
        for notification in notifications_to_send:
            if notification.get('match_id'):
                db.mark_notification_sent(notification['match_id'])
        
        return {
            "message": "Bildirimler gönderildi",
            "results": results,
            "job_title": job["title"],
            "company": job["company"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Bildirim gönderme hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/unsent-notifications")
async def get_unsent_notifications():
    """Henüz gönderilmemiş bildirimleri listele"""
    try:
        unsent_matches = db.get_unsent_matches()
        
        # Her eşleşme için iş ilanı ve aday bilgilerini ekle
        detailed_matches = []
        for match in unsent_matches:
            job = db.get_job_posting(match["job_id"])
            candidate = db.get_cv(match["candidate_id"])
            
            if job and candidate:
                candidate_email = candidate.get("cv_data", {}).get("contact_info", {}).get("email")
                match_detail = {
                    "match_id": match["_id"],
                    "job_title": job["title"],
                    "company": job["company"],
                    "candidate_name": candidate.get("cv_data", {}).get("names", ["Bilinmiyor"])[0] if candidate.get("cv_data", {}).get("names") else "Bilinmiyor",
                    "candidate_email": candidate_email,
                    "match_percentage": match["match_percentage"],
                    "created_at": match["created_at"]
                }
                detailed_matches.append(match_detail)
        
        return {
            "unsent_notifications": detailed_matches,
            "count": len(detailed_matches)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: str):
    """Aday kaydını sil"""
    try:
        candidate = db.get_cv(candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Aday bulunamadı")
        
        # GridFS'den dosyayı sil
        if candidate.get("file_id"):
            try:
                db.fs.delete(candidate["file_id"])
            except Exception as e:
                print(f"Dosya silme hatası: {e}")
        
        # Metadata'yı sil
        from bson import ObjectId
        result = db.cv_metadata.delete_one({"_id": ObjectId(candidate_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Aday silinirken hata oluştu")
        
        return {"message": "Aday başarıyla silindi"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/job-postings/{job_id}")
async def delete_job_posting(job_id: str):
    """İş ilanını sil"""
    try:
        from bson import ObjectId
        
        # İş ilanının varlığını kontrol et
        job = db.get_job_posting(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="İş ilanı bulunamadı")
        
        # İlişkili eşleşmeleri sil
        db.matches.delete_many({"job_id": job_id})
        
        # İş ilanını sil
        result = db.job_postings.delete_one({"_id": ObjectId(job_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="İş ilanı silinirken hata oluştu")
        
        return {"message": "İş ilanı ve ilişkili eşleşmeler başarıyla silindi"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Sistem istatistiklerini al"""
    try:
        candidates_count = len(db.get_all_candidates())
        jobs_count = len(db.get_all_job_postings())
        matches_count = db.matches.count_documents({})
        unsent_notifications_count = len(db.get_unsent_matches())
        
        return {
            "candidates": candidates_count,
            "job_postings": jobs_count,
            "total_matches": matches_count,
            "unsent_notifications": unsent_notifications_count,
            "system_status": "active"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)