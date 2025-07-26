from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from fastapi.encoders import jsonable_encoder

from cv_parse import EnhancedCVProcessor
from mongodb import Database
from mongodb import fix_mongo_ids
from notify import NotificationService
from vector import VectorMatcher


app = FastAPI(
    title="TalentMatch NLP",
    description="CV Analiz Uygulaması"
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

# Modeller
class JobPosting(BaseModel):
    title: str 
    description: str 
    requirements: List[str] 
    location: str
    company: str 
    matching_parameters: Optional[dict] = None  

class CandidateMatch(BaseModel):
    candidate_id: str  
    match_percentage: float 
    missing_skills: List[str] 
    explanation: str  

class MatchParameters(BaseModel):
    min_match_percentage: float = 70.0  
    required_skills: List[str] = []  
    preferred_skills: List[str] = [] 


@app.get("/")
async def root():
    return {"message": "TalentMatch NLP API"}

@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    """
    CV dosyası yükleme ve işleme (PDF/DOCX)
    """
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Sadece PDF ve DOCX dosyaları kabul edilir")

    try:
        # Dosya içeriğini oku
        file_content = await file.read()
        processor = EnhancedCVProcessor()
        result = processor.process_cv_file(file_content, file.filename)
        return {
            "message": result["message"],
            "status": result["status"],
            "file_id": result.get("file_id")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/job-posting")
async def create_job_posting(job: JobPosting):
    """
    Yeni iş ilanı oluşturma
    """
    try:
        job_id = db.store_job_posting(job.dict())
        return {
            "message": "İş ilanı başarıyla oluşturuldu",
            "job_id": job_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/match-candidates/{job_id}")
async def match_candidates(job_id: str):
    """
    Bir iş ilanı için uygun adayları bulma
    """
    try:
        # İş ilanını al
        job = db.get_job_posting(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="İş ilanı bulunamadı")
        # Tüm adayları al
        candidates = db.get_all_candidates()
        # Adayların summary'sini kullan
        for c in candidates:
            if "cv_data" in c and "summary" in c["cv_data"]:
                c["summary"] = c["cv_data"]["summary"]
            else:
                c["summary"] = ""
        vector_matcher.create_index(candidates)
        matches = vector_matcher.find_matches(
            f"{job['title']} {job['description']} {' '.join(job['requirements'])}"
        )
        for match in matches:
            match_id = db.store_match(job_id, match["candidate_id"], match)
            candidate = db.get_cv(match["candidate_id"])
            if candidate:
                notification_service.send_match_notification(
                    candidate.get("email"),
                    match
                )
        return fix_mongo_ids(matches)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/job-posting/{job_id}/parameters")
async def update_match_parameters(job_id: str, parameters: MatchParameters):
    """
    İş ilanı için eşleştirme parametrelerini güncelleme
    """
    try:
        success = db.update_match_parameters(job_id, parameters.dict())
        if not success:
            raise HTTPException(status_code=404, detail="İş ilanı bulunamadı")
        
        return {"message": "Eşleştirme parametreleri başarıyla güncellendi"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job-posting/{job_id}/matches")
async def get_job_matches(job_id: str):
    """
    Bir iş ilanı için tüm eşleşmeleri alma
    """
    try:
        matches = db.get_matches_for_job(job_id)
        return fix_mongo_ids(matches)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
