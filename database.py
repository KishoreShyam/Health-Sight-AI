import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthsight_history.db")

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    """Initializes the SQLite database and creates the patient_records table if missing."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            age INTEGER,
            duration REAL,
            pain_score REAL,
            lesion_size REAL,
            family_history TEXT,
            prediction_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            blob_url TEXT,
            azure_tags TEXT,
            speech_transcript TEXT,
            speech_language TEXT,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_record(
    patient_name="Anonymous Patient",
    age=50,
    duration=6.0,
    pain_score=3.0,
    lesion_size=10.0,
    family_history="No",
    prediction_label="Benign",
    confidence=92.8,
    risk_level="Low",
    blob_url=None,
    azure_tags=None,
    speech_transcript=None,
    speech_language="English"
):
    """Saves a new patient diagnostic prediction record into the database."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    def clean_text(val):
        if val is None:
            return ""
        if isinstance(val, bytes):
            return val.decode('utf-8', errors='ignore')
        return str(val)

    tags_str = ", ".join(azure_tags) if isinstance(azure_tags, list) else clean_text(azure_tags)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO patient_records (
            patient_name, age, duration, pain_score, lesion_size, family_history,
            prediction_label, confidence, risk_level, blob_url, azure_tags,
            speech_transcript, speech_language, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        clean_text(patient_name), int(age) if age else 50, float(duration) if duration else 6.0,
        float(pain_score) if pain_score else 3.0, float(lesion_size) if lesion_size else 10.0,
        clean_text(family_history), clean_text(prediction_label), float(confidence) if confidence else 0.0,
        clean_text(risk_level), clean_text(blob_url), tags_str,
        clean_text(speech_transcript), clean_text(speech_language), current_time
    ))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_all_records_df():
    """Retrieves all patient records as a pandas DataFrame ordered by latest timestamp."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM patient_records ORDER BY id DESC", conn)
    conn.close()
    return df

def get_record_by_id(record_id):
    """Retrieves a single patient record by ID."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_records WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        columns = ["id", "patient_name", "age", "duration", "pain_score", "lesion_size",
                   "family_history", "prediction_label", "confidence", "risk_level",
                   "blob_url", "azure_tags", "speech_transcript", "speech_language", "timestamp"]
        return dict(zip(columns, row))
    return None

def delete_record(record_id):
    """Deletes a record from the database."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_records WHERE id = ?", (record_id,))
    cursor.execute("DELETE FROM patient_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

# Auto-initialize database on import
init_db()
