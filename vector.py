import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import json
import re

class VectorMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Vektör eşleştiriciyi bir sentence transformer modeli ile başlat
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.candidates = []
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def create_index(self, candidates: List[Dict]):
        """
        Aday belgelerinden FAISS indeksi oluştur
        """
        self.candidates = candidates
        texts = [candidate.get("summary", "") for candidate in candidates]
        embeddings = self.model.encode(texts)
        
        # FAISS indeksini başlat
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
    def find_matches(self, query: str, k: int = 5) -> List[Dict]:
        """
        Bir sorgu için k en benzer adayı bul
        """
        if not self.index:
            raise ValueError("İndeks oluşturulmamış. Önce create_index'i çağırın.")
            
        # Sorguyu kodla
        query_vector = self.model.encode([query])[0]
        
        # FAISS indeksinde ara
        distances, indices = self.index.search(
            np.array([query_vector]).astype('float32'), k
        )
        
        # Sonuçları hazırla
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.candidates):
                candidate = self.candidates[idx]
                match_percentage = 100 * (1 - distance / 2)  # Mesafeyi yüzdeye çevir
                
                # Eksik becerileri bul
                missing_skills = self._find_missing_skills(
                    query, candidate.get("skills", [])
                )
                
                results.append({
                    "candidate_id": str(candidate.get("_id", "")),
                    "match_percentage": round(match_percentage, 2),
                    "missing_skills": missing_skills,
                    "explanation": self._generate_explanation(
                        match_percentage, missing_skills
                    )
                })
                
        return results
    
    def _find_missing_skills(self, query: str, candidate_skills: List[str]) -> List[str]:
        """
        Sorguda belirtilen ancak adayın becerilerinde olmayan becerileri bul
        """
        query_skills = set(re.findall(r'\b\w+\b', query.lower()))
        candidate_skills = set(skill.lower() for skill in candidate_skills)
        return list(query_skills - candidate_skills)
    
    def _generate_explanation(self, match_percentage: float, missing_skills: List[str]) -> str:
        """
        Eşleşme için açıklama oluştur
        """
        explanation = f"Eşleşme yüzdesi: {match_percentage}%"
        if missing_skills:
            explanation += f"\nEksik beceriler: {', '.join(missing_skills)}"
        return explanation
    
    def save_index(self, path: str):
        """
        FAISS indeksini diske kaydet
        """
        if self.index:
            faiss.write_index(self.index, path)
            
    def load_index(self, path: str):
        """
        FAISS indeksini diskten yükle
        """
        self.index = faiss.read_index(path) 