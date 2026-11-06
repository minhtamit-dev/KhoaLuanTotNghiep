import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'clinical_database.sqlite')

def get_db_connection(db_path=DB_PATH):
    """
    Tạo kết nối SQLite an toàn và BẬT ràng buộc khóa ngoại (Foreign Keys Enforcement).
    Khắc phục nhận xét: 'SQLite chưa bật ràng buộc khóa ngoại (PRAGMA foreign_keys)'.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    # BẮT BUỘC: Kích hoạt kiểm tra toàn vẹn khóa ngoại trong SQLite
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_database(db_path=DB_PATH):
    """
    Khởi tạo lược đồ cơ sở dữ liệu y tế chuẩn hóa tuân thủ bảo mật và toàn vẹn tham chiếu.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Bảng bệnh nhân (được ẩn danh hóa danh tính theo chuẩn y khoa HIPAA)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        anonymized_code TEXT UNIQUE NOT NULL,
        age_group TEXT DEFAULT 'Pediatric 1-5 yrs',
        gender TEXT DEFAULT 'Unknown',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. Bảng kết quả chẩn đoán từ mô hình AI (Ràng buộc Khóa ngoại tới patients)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS diagnoses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        pneumonia_probability REAL NOT NULL,
        prediction TEXT NOT NULL,
        classification_threshold REAL DEFAULT 0.5,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
    );
    """)
    
    # 3. Bảng thu thập nhận xét, đánh giá từ Bác sĩ Chẩn đoán hình ảnh (Ràng buộc Khóa ngoại tới diagnoses)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS physician_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        diagnosis_id INTEGER NOT NULL,
        physician_name TEXT NOT NULL,
        hospital_or_dept TEXT,
        agreement_status TEXT NOT NULL, -- 'Đồng ý', 'Không đồng ý', 'Cần hội chẩn'
        rating INTEGER CHECK (rating >= 1 AND rating <= 5),
        gradcam_fidelity_rating INTEGER CHECK (gradcam_fidelity_rating >= 1 AND gradcam_fidelity_rating <= 5),
        clinical_notes TEXT,
        reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id) ON DELETE CASCADE
    );
    """)
    
    conn.commit()
    conn.close()

def log_diagnosis(anonymized_code, model_name, prob, prediction, threshold=0.5, db_path=DB_PATH):
    """
    Ghi nhận một lượt chẩn đoán mới vào cơ sở dữ liệu an toàn.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR IGNORE INTO patients (anonymized_code) VALUES (?)", (anonymized_code,))
        cursor.execute("SELECT id FROM patients WHERE anonymized_code = ?", (anonymized_code,))
        patient_id = cursor.fetchone()['id']
        
        cursor.execute("""
        INSERT INTO diagnoses (patient_id, model_name, pneumonia_probability, prediction, classification_threshold)
        VALUES (?, ?, ?, ?, ?)
        """, (patient_id, model_name, float(prob), prediction, float(threshold)))
        
        diag_id = cursor.lastrowid
        conn.commit()
        return diag_id
    finally:
        conn.close()

def log_physician_review(diagnosis_id, doctor_name, department, agreement, rating, gradcam_rating, notes, db_path=DB_PATH):
    """
    Lưu ý kiến đánh giá từ Bác sĩ chẩn đoán hình ảnh.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO physician_reviews (diagnosis_id, physician_name, hospital_or_dept, agreement_status, rating, gradcam_fidelity_rating, clinical_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (diagnosis_id, doctor_name, department, agreement, int(rating), int(gradcam_rating), notes))
        conn.commit()
        return True
    except Exception as e:
        print(f"[!] Error logging physician review: {e}")
        return False
    finally:
        conn.close()

# Khởi tạo bảng ngay khi import
try:
    init_database()
except Exception:
    pass
