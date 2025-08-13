import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import json
import re
import logging

# Logging ayarlarını yapılandır
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Vektör eşleştiriciyi başlat
        
        Args:
            model_name: Kullanılacak sentence transformer modeli
        """
        try:
            logger.info(f"Sentence Transformer modeli yükleniyor: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model başarıyla yüklendi. Boyut: {self.dimension}")
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            raise Exception(f"Sentence Transformer modeli yüklenemedi: {e}")
        
        self.index = None
        self.candidates = []
        
    def create_index(self, candidates: List[Dict]):
        """
        Aday belgelerinden FAISS indeksi oluştur
        
        Args:
            candidates: Aday listesi, her aday 'summary' alanına sahip olmalı
        """
        if not candidates:
            raise ValueError("Aday listesi boş")
        
        try:
            self.candidates = candidates
            
            # Summary metinlerini hazırla
            texts = []
            for i, candidate in enumerate(candidates):
                summary = candidate.get("summary", "")
                
                # Summary boşsa diğer alanlardan metin oluştur
                if not summary or len(summary.strip()) < 10:
                    summary = self._create_fallback_text(candidate)
                
                texts.append(summary)
                logger.info(f"Aday {i+1}: {summary[:100]}...")
            
            # Metinleri vektörlere çevir
            logger.info(f"{len(texts)} aday için embedding oluşturuluyor...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # FAISS indeksini oluştur
            self.index = faiss.IndexFlatL2(self.dimension)
            embeddings_array = np.array(embeddings).astype('float32')
            
            # NaN değerlerini kontrol et
            if np.isnan(embeddings_array).any():
                logger.warning("Embedding'lerde NaN değerler tespit edildi, temizleniyor...")
                embeddings_array = np.nan_to_num(embeddings_array)
            
            self.index.add(embeddings_array)
            logger.info(f"FAISS indeksi oluşturuldu. Toplam aday: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"İndeks oluşturma hatası: {e}")
            raise Exception(f"Vektör indeksi oluşturulamadı: {e}")
    
    def _create_fallback_text(self, candidate: Dict) -> str:
        """Summary yoksa diğer alanlardan metin oluştur"""
        try:
            text_parts = []
            cv_data = candidate.get("cv_data", {})
            
            # İsimler
            names = cv_data.get("names", [])
            if names:
                text_parts.append(f"Ad: {names[0]}")
            
            # Beceriler
            skills = cv_data.get("skills", [])
            if skills:
                text_parts.append(f"Beceriler: {', '.join(skills[:10])}")
            
            # Deneyim
            experience = cv_data.get("experience", [])
            if experience:
                exp_texts = []
                for exp in experience[:3]:
                    if exp.get("position"):
                        exp_texts.append(exp["position"])
                    if exp.get("company"):
                        exp_texts.append(exp["company"])
                if exp_texts:
                    text_parts.append(f"Deneyim: {', '.join(exp_texts)}")
            
            # Eğitim
            education = cv_data.get("education", [])
            if education:
                edu_texts = []
                for edu in education[:2]:
                    if edu.get("institution"):
                        edu_texts.append(edu["institution"])
                    if edu.get("degree_type"):
                        edu_texts.append(edu["degree_type"])
                if edu_texts:
                    text_parts.append(f"Eğitim: {', '.join(edu_texts)}")
            
            fallback_text = "; ".join(text_parts)
            return fallback_text if fallback_text else "Detay bilgi bulunamadı"
            
        except Exception as e:
            logger.warning(f"Fallback metin oluşturma hatası: {e}")
            return "Aday bilgisi"
        
    def find_matches(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Dict]:
        """
        Bir sorgu için k en benzer adayı bul
        
        Args:
            query: Arama sorgusu (iş tanımı)
            k: Döndürülecek maksimum aday sayısı
            min_score: Minimum eşleşme skoru (0-100 arası)
            
        Returns:
            Eşleşen adayların listesi
        """
        if not self.index:
            raise ValueError("İndeks oluşturulmamış. Önce create_index'i çağırın.")
        
        if not query or len(query.strip()) < 3:
            raise ValueError("Sorgu çok kısa veya boş")
        
        try:
            # Sorguyu temizle ve hazırla
            cleaned_query = self._clean_query(query)
            logger.info(f"Temizlenmiş sorgu: {cleaned_query[:200]}...")
            
            # Sorguyu vektöre çevir
            query_vector = self.model.encode([cleaned_query])[0]
            
            # NaN kontrolü
            if np.isnan(query_vector).any():
                logger.warning("Sorgu vektöründe NaN değerler tespit edildi, temizleniyor...")
                query_vector = np.nan_to_num(query_vector)
            
            # FAISS indeksinde ara
            k_actual = min(k, self.index.ntotal)  # Mevcut aday sayısından fazla arama yapma
            distances, indices = self.index.search(
                np.array([query_vector]).astype('float32'), k_actual
            )
            
            # Sonuçları hazırla
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.candidates):
                    candidate = self.candidates[idx]
                    
                    # Mesafeyi yüzdeye çevir (0-1 arası normalize et)
                    # L2 mesafesi için: similarity = 1 / (1 + distance)
                    similarity = 1 / (1 + distance)
                    match_percentage = similarity * 100
                    
                    # Minimum skoru kontrol et
                    if match_percentage < min_score:
                        continue
                    
                    # Eksik becerileri bul
                    missing_skills = self._find_missing_skills(
                        query, candidate.get("cv_data", {}).get("skills", [])
                    )
                    
                    # Eşleşme açıklaması oluştur
                    explanation = self._generate_explanation(
                        match_percentage, missing_skills, candidate
                    )
                    
                    result = {
                        "candidate_id": str(candidate.get("_id", "")),
                        "match_percentage": round(match_percentage, 2),
                        "missing_skills": missing_skills,
                        "explanation": explanation,
                        "distance": float(distance)  # Debug için
                    }
                    
                    results.append(result)
                    
            # Sonuçları match_percentage'a göre sırala
            results.sort(key=lambda x: x["match_percentage"], reverse=True)
            
            logger.info(f"{len(results)} eşleşme bulundu")
            return results
            
        except Exception as e:
            logger.error(f"Eşleşme arama hatası: {e}")
            raise Exception(f"Eşleşme araması başarısız: {e}")
    
    def _clean_query(self, query: str) -> str:
        """Sorguyu temizle ve normalize et"""
        try:
            # Fazla boşlukları temizle
            query = re.sub(r'\s+', ' ', query)
            
            # HTML tag'lerini temizle (varsa)
            query = re.sub(r'<[^>]+>', '', query)
            
            # Gereksiz karakterleri temizle
            query = re.sub(r'[^\w\s.,;:-]', '', query)
            
            return query.strip()
        except Exception as e:
            logger.warning(f"Sorgu temizleme hatası: {e}")
            return query
    
    def _find_missing_skills(self, query: str, candidate_skills: List[str]) -> List[str]:
        """
        Sorguda belirtilen ancak adayın becerilerinde olmayan becerileri bul
        """
        try:
            # Sorgudan teknoloji/beceri anahtar kelimelerini çıkar
            tech_keywords = [
                'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js',
                'docker', 'kubernetes', 'aws', 'azure', 'git', 'sql', 'mongodb',
                'machine learning', 'ai', 'data science', 'html', 'css', 'php',
                'django', 'flask', 'spring', 'laravel', 'tensorflow', 'pytorch',
                'opencv', 'pandas', 'numpy', 'scikit-learn', 'tableau', 'powerbi'
            ]
            
            query_lower = query.lower()
            candidate_skills_lower = [skill.lower() for skill in candidate_skills]
            
            # Sorguda geçen ve adayda olmayan teknolojiler
            missing_skills = []
            for tech in tech_keywords:
                if tech in query_lower and tech not in ' '.join(candidate_skills_lower):
                    missing_skills.append(tech.title())
            
            # Tekrar eden skills'leri temizle
            missing_skills = list(set(missing_skills))
            
            # Maksimum 5 eksik beceri döndür
            return missing_skills[:5]
            
        except Exception as e:
            logger.warning(f"Eksik beceri analizi hatası: {e}")
            return []
    
    def _generate_explanation(self, match_percentage: float, missing_skills: List[str], candidate: Dict) -> str:
        """Eşleşme için detaylı açıklama oluştur"""
        try:
            explanation_parts = []
            
            # Ana eşleşme skoru
            if match_percentage >= 80:
                explanation_parts.append("Çok yüksek uyumluluk")
            elif match_percentage >= 60:
                explanation_parts.append("Yüksek uyumluluk")
            elif match_percentage >= 40:
                explanation_parts.append("Orta seviye uyumluluk")
            else:
                explanation_parts.append("Düşük uyumluluk")
            
            # Aday hakkında kısa bilgi
            cv_data = candidate.get("cv_data", {})
            
            # İsim
            names = cv_data.get("names", [])
            if names:
                explanation_parts.append(f"Aday: {names[0]}")
            
            # Temel beceriler
            skills = cv_data.get("skills", [])
            if skills:
                key_skills = skills[:3]
                explanation_parts.append(f"Ana beceriler: {', '.join(key_skills)}")
            
            # Deneyim özeti
            experience = cv_data.get("experience", [])
            if experience:
                recent_positions = [exp.get("position", "") for exp in experience[:2] if exp.get("position")]
                if recent_positions:
                    explanation_parts.append(f"Son pozisyonlar: {', '.join(recent_positions)}")
            
            # Eksik beceriler
            if missing_skills:
                explanation_parts.append(f"Geliştirilebilir alanlar: {', '.join(missing_skills[:3])}")
            
            return " | ".join(explanation_parts)
            
        except Exception as e:
            logger.warning(f"Açıklama oluşturma hatası: {e}")
            return f"Eşleşme oranı: %{match_percentage:.1f}"
    
    def save_index(self, path: str):
        """FAISS indeksini diske kaydet"""
        try:
            if self.index:
                faiss.write_index(self.index, path)
                logger.info(f"İndeks kaydedildi: {path}")
            else:
                raise ValueError("Kaydedilecek indeks bulunamadı")
        except Exception as e:
            logger.error(f"İndeks kaydetme hatası: {e}")
            raise Exception(f"İndeks kaydetme başarısız: {e}")
    
    def load_index(self, path: str):
        """FAISS indeksini diskten yükle"""
        try:
            self.index = faiss.read_index(path)
            logger.info(f"İndeks yüklendi: {path}")
        except Exception as e:
            logger.error(f"İndeks yükleme hatası: {e}")
            raise Exception(f"İndeks yükleme başarısız: {e}")
    
    def get_index_info(self) -> Dict:
        """İndeks hakkında bilgi döndür"""
        if not self.index:
            return {"status": "no_index", "total_candidates": 0}
        
        return {
            "status": "ready",
            "total_candidates": self.index.ntotal,
            "dimension": self.dimension,
            "model_name": self.model._modules['0'].auto_model.name_or_path if hasattr(self.model, '_modules') else "unknown"
        }
    
    def batch_search(self, queries: List[str], k: int = 5) -> List[List[Dict]]:
        """
        Birden fazla sorgu için toplu arama yap
        
        Args:
            queries: Sorgu listesi
            k: Her sorgu için döndürülecek aday sayısı
            
        Returns:
            Her sorgu için eşleşme listesi
        """
        if not self.index:
            raise ValueError("İndeks oluşturulmamış")
        
        try:
            results = []
            for query in queries:
                matches = self.find_matches(query, k)
                results.append(matches)
            
            logger.info(f"{len(queries)} sorgu için toplu arama tamamlandı")
            return results
            
        except Exception as e:
            logger.error(f"Toplu arama hatası: {e}")
            raise Exception(f"Toplu arama başarısız: {e}")
    
    def update_candidate(self, candidate_id: str, new_candidate_data: Dict):
        """
        Belirli bir adayın bilgilerini güncelle ve indeksi yeniden oluştur
        
        Args:
            candidate_id: Güncellenecek adayın ID'si
            new_candidate_data: Yeni aday verisi
        """
        try:
            # Mevcut adayları güncelle
            for i, candidate in enumerate(self.candidates):
                if str(candidate.get("_id", "")) == candidate_id:
                    self.candidates[i] = new_candidate_data
                    break
            
            # İndeksi yeniden oluştur
            self.create_index(self.candidates)
            logger.info(f"Aday {candidate_id} güncellendi ve indeks yenilendi")
            
        except Exception as e:
            logger.error(f"Aday güncelleme hatası: {e}")
            raise Exception(f"Aday güncelleme başarısız: {e}")
    
    def remove_candidate(self, candidate_id: str):
        """
        Belirli bir adayı kaldır ve indeksi yeniden oluştur
        
        Args:
            candidate_id: Kaldırılacak adayın ID'si
        """
        try:
            # Adayı listeden çıkar
            original_count = len(self.candidates)
            self.candidates = [c for c in self.candidates if str(c.get("_id", "")) != candidate_id]
            
            if len(self.candidates) < original_count:
                # İndeksi yeniden oluştur
                if self.candidates:
                    self.create_index(self.candidates)
                else:
                    self.index = None
                
                logger.info(f"Aday {candidate_id} kaldırıldı ve indeks güncellendi")
            else:
                logger.warning(f"Aday {candidate_id} bulunamadı")
                
        except Exception as e:
            logger.error(f"Aday kaldırma hatası: {e}")
            raise Exception(f"Aday kaldırma başarısız: {e}")