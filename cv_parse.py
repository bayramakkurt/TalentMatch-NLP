from docx import Document
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import re
import spacy
import gridfs
import pymongo
from pymongo import MongoClient
from bson import ObjectId
import io
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
import pdfplumber
from mongodb import Database

# spaCy modelini yükle
nlp = spacy.load("tr_core_news_trf")
db_manager = Database()

@dataclass
class CVInfo:
    names: List[str]
    education: List[Dict[str, str]]
    experience: List[Dict[str, str]]
    skills: List[str]
    contact_info: Dict[str, str]
    summary: str = ""


class CVSummarizer:
    """CV özetleme sınıfı - Türkçe T5 modeli kullanır"""
    
    def __init__(self, model_name: str = "ozcangundes/mt5-small-turkish-summarization"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Modeli ve tokenizer'ı yükle"""
        try:
            self.tokenizer = MT5Tokenizer.from_pretrained(self.model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name)
            
            # GPU varsa kullan
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            self.tokenizer = None
            self.model = None
    
    def summarize_cv(self, cv_text: str, max_length: int = 200, min_length: int = 50) -> str:
        """CV metnini özetle"""
        if not self.model or not self.tokenizer:
            return "Model yüklenemedi, özet oluşturulamadı."
        
        try:
            # Metni temizle ve hazırla
            cleaned_text = self._prepare_text_for_summarization(cv_text)
            
            # Tokenize et
            inputs = self.tokenizer.encode(
                cleaned_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # GPU'ya taşı
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Özet oluştur
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode et
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary.strip()
            
        except Exception as e:
            print(f"Özetleme hatası: {e}")
            return "Özet oluşturulamadı."
    
    def _prepare_text_for_summarization(self, text: str) -> str:
        """Metni özetleme için hazırla"""
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        
        # Çok kısa satırları birleştir
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Çok kısa satırları atla
                cleaned_lines.append(line)
        
        # İlk 2000 karakteri al (model limitine uygun)
        cleaned_text = ' '.join(cleaned_lines)[:2000]
        
        return cleaned_text
    

class CVExtractor:
    def __init__(self):
        # Eğitim anahtar kelimeleri (Türkçe-İngilizce)
        self.education_keywords = [
            'üniversite', 'university', 'lisans', 'bachelor', 'yüksek lisans', 'master',
            'doktora', 'phd', 'doctorate', 'lise', 'high school', 'kolej', 'college',
            'okul', 'school', 'fakülte', 'faculty', 'bölüm', 'department', 'mezun',
            'graduate', 'graduated', 'derece', 'degree', 'diploma', 'sertifika',
            'certificate', 'kurs', 'course', 'eğitim', 'education', 'öğrenim', 'study',
            'akademi', 'academy', 'enstitü', 'institute', 'meslek yüksekokulu',
            'vocational school', 'teknik', 'technical'
        ]
        
        # Deneyim anahtar kelimeleri (Türkçe-İngilizce)
        self.experience_keywords = [
            'deneyim', 'experience', 'tecrübe', 'çalışma', 'work', 'kariyer', 'career',
            'pozisyon', 'position', 'görev', 'role', 'iş', 'job', 'şirket', 'company',
            'firma', 'corporation', 'kurum', 'institution', 'işyeri', 'workplace',
            'staj', 'internship', 'proje', 'project', 'sorumlu', 'responsible',
            'müdür', 'manager', 'uzman', 'specialist', 'geliştirici', 'developer',
            'mühendis', 'engineer', 'analyst', 'analist', 'koordinatör', 'coordinator'
        ]
        
        # Skill anahtar kelimeleri
        self.skill_keywords = [
            'yetenekler', 'skills', 'beceriler', 'competencies', 'abilities',
            'yetenek', 'skill', 'beceri', 'teknoloji', 'technology', 'araçlar',
            'tools', 'yazılım', 'software', 'diller', 'languages', 'programlama',
            'programming', 'teknik', 'technical'
        ]
        
        # Tarih pattern'leri (düzeltilmiş)
        self.date_patterns = [
            r'\d{4}[-/]\d{4}',  # 2020-2024
            r'\d{4}\s*-\s*\d{4}',  # 2020 - 2024
            r'\d{1,2}[./]\d{4}',  # 01.2020
            r'[A-Za-z]+\s+\d{4}',  # Ocak 2020, January 2020
            r'\d{4}\s*-\s*[Hh]alen',  # 2020 - Halen
            r'\d{4}\s*-\s*[Pp]resent',  # 2020 - Present
            r'\d{4}\s*-\s*[Dd]evam',  # 2020 - Devam
            r'\d{4}\s*-\s*[Gg]ünümüz',  # 2020 - Günümüz
        ]

    def extract_names(self, text: str) -> List[str]:
        """İsim soyisim çıkarma - NER + sıkı regex ile"""
        doc = nlp(text)
        names = []

        # NER ile PERSON entityleri
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                # En az 2 kelime ve sadece harf içeren
                if len(name.split()) >= 2 and all(re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", w) for w in name.split()):
                    names.append(name)

        # Başından sıkı regex ile 2-4 kelimelik isim arama
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            # Her kelime büyük harfle başlar, 2-4 kelime, sadece harflerden oluşur
            if re.fullmatch(r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s|$)){2,4}', line):
                if line not in names:
                    names.append(line)

        return list(set(names))

    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Eğitim bilgilerini çıkarma - yapılandırılmış parsing"""
        education = []
        
        # Önce eğitim bölümlerini bul
        education_sections = self._find_education_sections(text)
        
        for section in education_sections:
            edu_entries = self._parse_education_entries(section)
            education.extend(edu_entries)
        
        # Eğer bölüm bulunamazsa, satır satır ara
        if not education:
            education = self._extract_education_line_by_line(text)
        
        return education
    
    def _find_education_sections(self, text: str) -> List[str]:
        """Eğitim bölümlerini tespit et"""
        sections = []
        lines = text.split('\n')
        
        education_section_headers = [
            'eğitim', 'education', 'öğrenim', 'akademik', 'academic',
            'eğitim bilgileri', 'educational background', 'qualifications'
        ]
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            # Başlık satırı mı kontrol et
            if (any(header in line_clean for header in education_section_headers) and 
                len(line_clean.split()) <= 3):
                
                # Bu bölümün içeriğini topla
                section_content = []
                j = i + 1
                
                while j < len(lines):
                    next_line = lines[j].strip()
                    
                    # Boş satır geç
                    if not next_line:
                        j += 1
                        continue
                    
                    # Yeni bölüm başladı mı?
                    if self._is_new_section_header(next_line):
                        break
                    
                    section_content.append(next_line)
                    j += 1
                    
                    # Maksimum 15 satır al
                    if len(section_content) >= 15:
                        break
                
                if section_content:
                    sections.append('\n'.join(section_content))
        
        return sections
    
    def _parse_education_entries(self, section_text: str) -> List[Dict[str, str]]:
        """Eğitim bölümünden entry'leri çıkar"""
        entries = []
        lines = section_text.split('\n')
        
        current_entry = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Bu satır yeni bir eğitim girişi mi?
            if self._is_education_entry(line):
                # Önceki entry'i kaydet
                if current_entry:
                    entries.append(current_entry)
                
                # Yeni entry başlat
                current_entry = self._parse_single_education_line(line)
            
            # Mevcut entry'e detay ekle
            elif current_entry and self._is_education_detail(line):
                if 'details' not in current_entry:
                    current_entry['details'] = []
                current_entry['details'].append(line)
        
        # Son entry'i kaydet
        if current_entry:
            entries.append(current_entry)
        
        return entries
    
    def _is_education_entry(self, line: str) -> bool:
        """Bu satır bir eğitim girişi mi?"""
        line_lower = line.lower()
        
        # Tarih içeriyorsa büyük ihtimalle ana giriş
        has_date = any(re.search(pattern, line) for pattern in self.date_patterns)
        
        # Eğitim kurumu içeriyorsa
        has_institution = bool(self._extract_institution_name(line))
        
        # Derece türü içeriyorsa
        has_degree = any(word in line_lower for word in ['lisans', 'bachelor', 'master', 'yüksek lisans', 'doktora', 'phd', 'lise'])
        
        return has_date or has_institution or has_degree
    
    def _is_education_detail(self, line: str) -> bool:
        """Bu satır eğitim detayı mı?"""
        line_lower = line.lower()
        
        # Not ortalaması, tez konusu gibi detaylar
        detail_keywords = [
            'gpa', 'not ortalaması', 'ortalama', 'tez', 'thesis', 'proje', 'project',
            'burs', 'scholarship', 'derece', 'honor', 'onur', 'başarı', 'achievement'
        ]
        
        return any(keyword in line_lower for keyword in detail_keywords)
    
    def _parse_single_education_line(self, line: str) -> Dict[str, str]:
        """Tek satırlık eğitim bilgisini parse et"""
        edu_info = {}
        line_lower = line.lower()
        
        # Tarih çıkar
        for pattern in self.date_patterns:
            try:
                dates = re.findall(pattern, line)
                if dates:
                    edu_info['dates'] = dates[0]
                    # Tarihi çıkardıktan sonra temizle
                    line = re.sub(pattern, '', line).strip()
                    break
            except re.error:
                continue
        
        # Kurum ismi
        institution = self._extract_institution_name(line)
        if institution:
            edu_info['institution'] = institution
        
        # Bölüm
        department = self._extract_department(line)
        if department:
            edu_info['department'] = department
        
        # Derece türü
        if any(word in line_lower for word in ['lisans', 'bachelor']):
            edu_info['degree_type'] = 'Lisans'
        elif any(word in line_lower for word in ['yüksek lisans', 'master']):
            edu_info['degree_type'] = 'Yüksek Lisans'
        elif any(word in line_lower for word in ['doktora', 'phd']):
            edu_info['degree_type'] = 'Doktora'
        elif any(word in line_lower for word in ['lise', 'high school']):
            edu_info['degree_type'] = 'Lise'
        elif any(word in line_lower for word in ['önlisans', 'associate']):
            edu_info['degree_type'] = 'Önlisans'
        
        return edu_info
    
    def _extract_education_line_by_line(self, text: str) -> List[Dict[str, str]]:
        """Satır satır eğitim arama (fallback)"""
        education = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Bu satırda eğitim bilgisi var mı?
            if (any(keyword in line_lower for keyword in ['üniversite', 'university', 'lise', 'college']) and
                (any(re.search(pattern, line) for pattern in self.date_patterns) or 
                 any(word in line_lower for word in ['lisans', 'bachelor', 'master', 'phd']))):
                
                edu_info = self._parse_single_education_line(line)
                education.append(edu_info)
        
        return education

    def _extract_institution_name(self, text: str) -> str:
        """Kurum ismi çıkarma"""
        text_lower = text.lower()
        
        # Üniversite pattern'leri
        university_patterns = [
            r'([A-ZÜĞŞIÖÇa-züğşiöç\s]+)\s*üniversitesi',  # ... Üniversitesi
            r'([A-ZÜĞŞIÖÇa-züğşiöç\s]+)\s*university',    # ... University
            r'([A-ZÜĞŞIÖÇa-züğşiöç\s]+)\s*üniversite',    # ... Üniversite
            r'([A-ZÜĞŞIÖÇa-züğşiöç\s]+)\s*lisesi',        # ... Lisesi
            r'([A-ZÜĞŞIÖÇa-züğşiöç\s]+)\s*college',       # ... College
            r'([A-ZÜĞŞIÖÇa-züğşiöç\s]+)\s*institute',     # ... Institute
            r'([A-ZÜĞŞIÖÇa-züğşiöç\s]+)\s*akademi',       # ... Akademi
        ]
        
        for pattern in university_patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    institution = match.group(1).strip()
                    # En az 3 karakter olmalı
                    if len(institution) >= 3:
                        return institution
            except re.error:
                continue
        
        return None

    def _extract_department(self, text: str) -> str:
        """Bölüm/alan çıkarma"""
        # Yaygın bölüm isimleri
        departments = [
            'bilgisayar mühendisliği', 'computer engineering', 'computer science',
            'elektrik mühendisliği', 'electrical engineering',
            'makine mühendisliği', 'mechanical engineering',
            'endüstri mühendisliği', 'industrial engineering',
            'işletme', 'business administration', 'management',
            'ekonomi', 'economics', 'iktisat',
            'hukuk', 'law', 'hukuk fakültesi',
            'tıp', 'medicine', 'medical',
            'psikoloji', 'psychology',
            'matematik', 'mathematics',
            'fizik', 'physics',
            'kimya', 'chemistry',
            'biyoloji', 'biology',
            'mimarlık', 'architecture'
        ]
        
        text_lower = text.lower()
        for dept in departments:
            if dept in text_lower:
                return dept.title()
        
        return None

    def extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Deneyim ve tecrübe bilgilerini çıkarma - yapılandırılmış parsing"""
        experience = []
        
        # Önce deneyim bölümlerini bul
        experience_sections = self._find_experience_sections(text)
        
        for section in experience_sections:
            exp_entries = self._parse_experience_entries(section)
            experience.extend(exp_entries)
        
        # Eğer bölüm bulunamazsa, satır satır ara
        if not experience:
            experience = self._extract_experience_line_by_line(text)
        
        return experience
    
    def _find_experience_sections(self, text: str) -> List[str]:
        """Deneyim bölümlerini tespit et"""
        sections = []
        lines = text.split('\n')
        
        experience_section_headers = [
            'deneyim', 'experience', 'tecrübe', 'iş deneyimi', 'work experience',
            'kariyer', 'career', 'professional experience', 'çalışma geçmişi',
            'employment', 'employment history', 'work history'
        ]
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            # Başlık satırı mı kontrol et
            if (any(header in line_clean for header in experience_section_headers) and 
                len(line_clean.split()) <= 4):
                
                # Bu bölümün içeriğini topla
                section_content = []
                j = i + 1
                
                while j < len(lines):
                    next_line = lines[j].strip()
                    
                    # Boş satır geç
                    if not next_line:
                        j += 1
                        continue
                    
                    # Yeni bölüm başladı mı?
                    if self._is_new_section_header(next_line):
                        break
                    
                    section_content.append(next_line)
                    j += 1
                    
                    # Maksimum 20 satır al
                    if len(section_content) >= 20:
                        break
                
                if section_content:
                    sections.append('\n'.join(section_content))
        
        return sections
    
    def _parse_experience_entries(self, section_text: str) -> List[Dict[str, str]]:
        """Deneyim bölümünden entry'leri çıkar"""
        entries = []
        lines = section_text.split('\n')
        
        current_entry = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Bu satır yeni bir deneyim girişi mi?
            if self._is_experience_entry(line):
                # Önceki entry'i kaydet
                if current_entry:
                    entries.append(current_entry)
                
                # Yeni entry başlat
                current_entry = self._parse_single_experience_line(line)
            
            # Mevcut entry'e açıklama/sorumluluk ekle
            elif current_entry and self._is_experience_description(line):
                if 'responsibilities' not in current_entry:
                    current_entry['responsibilities'] = []
                current_entry['responsibilities'].append(line)
        
        # Son entry'i kaydet
        if current_entry:
            entries.append(current_entry)
        
        return entries
    
    def _is_experience_entry(self, line: str) -> bool:
        """Bu satır bir deneyim girişi mi?"""
        line_lower = line.lower()
        
        # Tarih içeriyorsa büyük ihtimalle ana giriş
        has_date = any(re.search(pattern, line) for pattern in self.date_patterns)
        
        # Pozisyon title içeriyorsa
        position_indicators = [
            'manager', 'müdür', 'developer', 'geliştirici', 'engineer', 'mühendis',
            'analyst', 'analist', 'specialist', 'uzman', 'coordinator', 'koordinatör',
            'assistant', 'asistan', 'intern', 'stajyer', 'lead', 'senior', 'junior',
            'director', 'direktör', 'supervisor', 'süpervizör', 'consultant', 'danışman'
        ]
        has_position = any(pos in line_lower for pos in position_indicators)
        
        # Şirket ismi pattern'i (genelde büyük harflerle)
        has_company_pattern = bool(re.search(r'[A-ZÜĞŞIÖÇa-z][A-ZÜĞŞIÖÇa-z\s&,.-]{3,}', line))
        
        # Çizgi (-) veya tire ile ayrılmış format (Pozisyon - Şirket)
        has_separator = ' - ' in line or ' | ' in line
        
        return has_date or (has_position and has_company_pattern) or has_separator
    
    def _is_experience_description(self, line: str) -> bool:
        """Bu satır deneyim açıklaması mı?"""
        line_lower = line.lower()
        
        # Bullet point karakterleri
        if line.startswith(('•', '-', '*', '○', '►', '▪')):
            return True
        
        # Sorumluluk belirten kelimeler
        responsibility_keywords = [
            'sorumlu', 'responsible', 'geliştir', 'develop', 'yönet', 'manage',
            'analiz', 'analyze', 'tasarla', 'design', 'uygula', 'implement',
            'koordine', 'coordinate', 'liderlik', 'lead', 'organize', 'organize',
            'proje', 'project', 'takım', 'team'
        ]
        
        # Açıklama satırı genelde uzundur ve fiil içerir
        if len(line) > 20 and any(keyword in line_lower for keyword in responsibility_keywords):
            return True
        
        return False
    
    def _parse_single_experience_line(self, line: str) -> Dict[str, str]:
        """Tek satırlık deneyim bilgisini parse et"""
        exp_info = {}
        line_lower = line.lower()
        original_line = line
        
        # Tarih çıkar
        for pattern in self.date_patterns:
            try:
                dates = re.findall(pattern, line)
                if dates:
                    exp_info['dates'] = dates[0]
                    # Tarihi çıkardıktan sonra temizle
                    line = re.sub(pattern, '', line).strip()
                    break
            except re.error:
                continue
        
        # Pozisyon ve şirket ayırma
        # Format 1: Pozisyon - Şirket
        if ' - ' in line:
            parts = line.split(' - ', 1)
            if len(parts) == 2:
                exp_info['position'] = parts[0].strip()
                exp_info['company'] = parts[1].strip()
        
        # Format 2: Pozisyon | Şirket
        elif ' | ' in line:
            parts = line.split(' | ', 1)
            if len(parts) == 2:
                exp_info['position'] = parts[0].strip()
                exp_info['company'] = parts[1].strip()
        
        # Format 3: Pozisyon, Şirket
        elif ', ' in line and len(line.split(', ')) == 2:
            parts = line.split(', ')
            # Pozisyon genelde daha kısa olur
            if len(parts[0]) < len(parts[1]):
                exp_info['position'] = parts[0].strip()
                exp_info['company'] = parts[1].strip()
            else:
                exp_info['company'] = parts[0].strip()
                exp_info['position'] = parts[1].strip()
        
        else:
            # Tek satırda hem pozisyon hem şirket var, ayırmaya çalış
            position_keywords = [
                'manager', 'müdür', 'developer', 'geliştirici', 'engineer', 'mühendis',
                'analyst', 'analist', 'specialist', 'uzman', 'intern', 'stajyer'
            ]
            
            # Pozisyon keyword'ü bul
            for keyword in position_keywords:
                if keyword in line_lower:
                    # Keyword'den öncesi ve sonrası
                    keyword_pos = line_lower.find(keyword)
                    before = line[:keyword_pos].strip()
                    after = line[keyword_pos:].strip()
                    
                    # Daha uzun olan şirket ismi olabilir
                    if len(before) > len(after):
                        exp_info['company'] = before
                        exp_info['position'] = after
                    else:
                        exp_info['position'] = before + ' ' + after
                    break
            else:
                # Ayırt edemediyse tümünü position olarak al
                exp_info['position'] = line.strip()
        
        return exp_info
    
    def _extract_experience_line_by_line(self, text: str) -> List[Dict[str, str]]:
        """Satır satır deneyim arama (fallback)"""
        experience = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Bu satırda deneyim bilgisi var mı?
            if (any(keyword in line_lower for keyword in self.experience_keywords) and
                (any(re.search(pattern, line) for pattern in self.date_patterns) or
                 any(pos in line_lower for pos in ['manager', 'developer', 'engineer', 'analyst']))):
                
                exp_info = self._parse_single_experience_line(line)
                experience.append(exp_info)
        
        return experience
    
    def _is_new_section_header(self, line: str) -> bool:
        """Bu satır yeni bölüm başlığı mı?"""
        line_lower = line.strip().lower()
        
        section_headers = [
            'eğitim', 'education', 'deneyim', 'experience', 'yetenekler', 'skills',
            'projeler', 'projects', 'sertifika', 'certificates', 'referans', 'references',
            'iletişim', 'contact', 'kişisel', 'personal', 'özet', 'summary',
            'hobiler', 'hobbies', 'dil', 'languages'
        ]
        
        # Kısa ve bölüm başlığı içeren satırlar
        return (len(line_lower.split()) <= 3 and 
                any(header in line_lower for header in section_headers))

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        contact = {}

        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact['email'] = emails[0]

        phone_patterns = [
            r'\+90\s*\(?\d{3}\)?\s*\d{3}\s*\d{2}\s*\d{2}',
            r'0\d{3}\s?\d{3}\s?\d{2}\s?\d{2}',
            r'\(?0\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}',
            r'\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}',
        ]

        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                contact['phone'] = phones[0].strip()
                break

        return contact

    def extract_skills(self, text: str) -> List[str]:
        """Yetenek ve beceri bilgilerini çıkarma - dinamik skill tespiti"""
        skills = []
        text_lower = text.lower()
        
        # Programlama dilleri ve teknolojiler (genişletilmiş liste)
        tech_skills = [
            # Programlama dilleri
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
            'go', 'rust', 'kotlin', 'swift', 'dart', 'scala', 'perl', 'r', 'matlab',
            
            # Web teknolojileri
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django',
            'flask', 'spring', 'laravel', 'bootstrap', 'jquery', 'sass', 'less',
            
            # Veritabanları
            'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis',
            'elasticsearch', 'cassandra', 'neo4j',
            
            # Cloud ve DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'gitlab',
            'github', 'terraform', 'ansible', 'chef', 'puppet',
            
            # Data Science & AI
            'machine learning', 'makine öğrenmesi', 'deep learning', 'ai', 'yapay zeka',
            'data science', 'veri bilimi', 'pandas', 'numpy', 'scikit-learn', 'tensorflow',
            'pytorch', 'keras', 'opencv', 'nlp', 'computer vision',
            
            # Analiz araçları
            'excel', 'powerbi', 'tableau', 'qlik', 'spss', 'sas', 'stata',
            
            # Tasarım
            'photoshop', 'illustrator', 'figma', 'sketch', 'indesign', 'after effects',
            'premiere', 'autocad', 'solidworks', 'blender', '3ds max',
            
            # Metodolojiler
            'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'ci/cd',
            
            # Soft skills
            'leadership', 'liderlik', 'communication', 'iletişim', 'teamwork', 
            'takım çalışması', 'problem solving', 'problem çözme', 'analytical',
            'analitik', 'creative', 'yaratıcı', 'adaptable', 'uyum'
        ]
        
        # Metin içinde skill arama
        for skill in tech_skills:
            if skill in text_lower:
                skills.append(skill.title())
        
        # Skill bölümlerini dinamik olarak bul
        skill_sections = self._find_skill_sections(text)
        
        for section in skill_sections:
            section_skills = self._extract_skills_from_section(section)
            skills.extend(section_skills)
        
        return list(set(skills))

    def _find_skill_sections(self, text: str) -> List[str]:
        """Skill bölümlerini bulma"""
        sections = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Skill başlığı kontrolü
            if any(keyword in line_lower for keyword in self.skill_keywords):
                # Bu satırdan sonraki satırları topla (yeni bölüm başlayana kadar)
                section_lines = []
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    # Yeni bölüm başladı mı kontrol et
                    if (any(keyword in next_line.lower() for keyword in self.education_keywords + self.experience_keywords) 
                        and len(next_line.split()) <= 3):
                        break
                    
                    section_lines.append(next_line)
                    
                    # Maksimum 10 satır al
                    if len(section_lines) >= 10:
                        break
                
                if section_lines:
                    sections.append(' '.join(section_lines))
        
        return sections

    def _extract_skills_from_section(self, section_text: str) -> List[str]:
        """Bir bölümden skill çıkarma"""
        skills = []
        
        # Virgül, nokta, satır sonu ile ayrılmış skill'leri bul
        delimiters = r'[,\n•\-\*\|/\\]'
        potential_skills = re.split(delimiters, section_text)
        
        for skill in potential_skills:
            skill = skill.strip()
            # Uygun uzunlukta ve anlamlı skill'leri al
            if 2 < len(skill) < 50 and not skill.isdigit():
                # Fazla noktalama işareti içermiyorsa
                if skill.count('.') < 2 and skill.count('(') < 2:
                    skills.append(skill)
        
        return skills

    def extract_cv_info(self, cv_text: str) -> CVInfo:
        """Ana fonksiyon - CV'den tüm bilgileri çıkarır"""
        # Metni temizle
        cv_text = re.sub(r'\s+', ' ', cv_text)  # Fazla boşlukları temizle
        
        # Tüm bilgileri çıkar
        names = self.extract_names(cv_text)
        education = self.extract_education(cv_text)
        experience = self.extract_experience(cv_text)
        skills = self.extract_skills(cv_text)
        contact_info = self.extract_contact_info(cv_text)
        
        return CVInfo(
            names=names,
            education=education,
            experience=experience,
            skills=skills,
            contact_info=contact_info
        )

class EnhancedCVProcessor:
    """Ana CV işleme sınıfı - tüm işlemleri koordine eder"""
    
    def __init__(self):
        self.extractor = CVExtractor()
        self.summarizer = CVSummarizer()
        
    
    def pdf_to_text(self, pdf_content: bytes) -> str:
        """PDF içeriğini metne çevir (pdfplumber ile)"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                return full_text
        except Exception as e:
            raise Exception(f"PDF okuma hatası: {e}")
    
    def doc_to_text(self, doc_content: bytes) -> str:
        """DOC/DOCX içeriğini metne çevir"""
        try:
            doc_stream = io.BytesIO(doc_content)
            doc = Document(doc_stream)
            full_text = ""
            
            for para in doc.paragraphs:
                full_text += para.text + "\n"
            
            return full_text
        except Exception as e:
            raise Exception(f"DOC okuma hatası: {e}")
    
    def process_cv_file(self, file_content: bytes, filename: str, content_type: str = None) -> Dict[str, Any]:
        """
        Ana fonksiyon - CV dosyasını işler
        
        Args:
            file_content: Dosya içeriği (bytes)
            filename: Dosya adı
            content_type: MIME type (isteğe bağlı)
            
        Returns:
            Dict: İşlenmiş CV verileri ve metadata
        """
        
        try:
            # 1. Dosya tipini belirle
            if content_type is None:
                if filename.lower().endswith('.pdf'):
                    content_type = 'application/pdf'
                elif filename.lower().endswith(('.doc', '.docx')):
                    content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                else:
                    raise ValueError("Desteklenmeyen dosya formatı. PDF veya DOC/DOCX dosyası gerekli.")
            
            # 2. Dosyayı MongoDB GridFS'e kaydet
            file_id = db_manager.save_cv_file(file_content, filename, content_type)
            
            # 3. Dosya içeriğini metne çevir
            if 'pdf' in content_type.lower():
                cv_text = self.pdf_to_text(file_content)
            elif 'word' in content_type.lower() or 'document' in content_type.lower():
                cv_text = self.doc_to_text(file_content)
            else:
                raise ValueError("Desteklenmeyen dosya formatı")
            
            # 4. CV bilgilerini çıkar
            cv_info = self.extractor.extract_cv_info(cv_text)
            
            # 5. CV özetini oluştur
            cv_summary = self.summarizer.summarize_cv(cv_text)
            cv_info.summary = cv_summary
            
            # 6. CV metadata'sını MongoDB'ye kaydet
            metadata_id = db_manager.save_cv_metadata(file_id, cv_info, filename)
            
            # 7. Sonucu döndür
            result = {
                'file_id': file_id,
                'metadata_id': metadata_id,
                'filename': filename,
                'content_type': content_type,
                'cv_data': {
                    'names': cv_info.names,
                    'education': cv_info.education,
                    'experience': cv_info.experience,
                    'skills': cv_info.skills,
                    'contact_info': cv_info.contact_info,
                    'summary': cv_info.summary
                },
                'status': 'success',
                'message': 'CV başarıyla işlendi ve kaydedildi'
            }
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'CV işleme hatası: {str(e)}',
                'cv_data': None
            }