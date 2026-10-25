import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(__file__), "rehabvision.db")

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS exercises (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        target_joint TEXT NOT NULL,
        description TEXT,
        target_rom_min REAL,
        target_rom_max REAL,
        created_at TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS references_table (
        id TEXT PRIMARY KEY,
        exercise_id TEXT NOT NULL,
        exercise_name TEXT NOT NULL,
        notes TEXT,
        duration_seconds REAL NOT NULL,
        frame_count INTEGER NOT NULL,
        joint_angles TEXT NOT NULL,
        landmarks_summary TEXT NOT NULL,
        created_at TEXT NOT NULL,
        is_demo INTEGER DEFAULT 0,
        reference_type TEXT DEFAULT 'standard',
        version TEXT DEFAULT '1.0',
        provenance TEXT DEFAULT 'Clinical Protocol Baseline',
        patient_id TEXT DEFAULT NULL,
        is_active INTEGER DEFAULT 1
    )
    """)

    # Ensure migration columns for existing databases
    for col_name, col_type in [
        ("reference_type", "TEXT DEFAULT 'standard'"),
        ("version", "TEXT DEFAULT '1.0'"),
        ("provenance", "TEXT DEFAULT 'Clinical Protocol Baseline'"),
        ("patient_id", "TEXT DEFAULT NULL"),
        ("is_active", "INTEGER DEFAULT 1"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE references_table ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        exercise_id TEXT NOT NULL,
        exercise_name TEXT NOT NULL,
        reference_id TEXT,
        repetitions_completed INTEGER NOT NULL,
        duration_seconds REAL NOT NULL,
        angle_stats TEXT NOT NULL,
        comparison_score REAL,
        joint_deviations TEXT NOT NULL,
        model_status TEXT NOT NULL,
        lstm_prediction TEXT,
        warnings TEXT NOT NULL,
        notes TEXT,
        raw_trajectory TEXT,
        is_demo INTEGER DEFAULT 0,
        reference_type TEXT DEFAULT 'standard',
        reference_version TEXT DEFAULT '1.0',
        reference_provenance TEXT DEFAULT NULL,
        patient_id TEXT DEFAULT NULL
    )
    """)

    for col_name, col_type in [
        ("reference_type", "TEXT DEFAULT 'standard'"),
        ("reference_version", "TEXT DEFAULT '1.0'"),
        ("reference_provenance", "TEXT DEFAULT NULL"),
        ("patient_id", "TEXT DEFAULT NULL"),
        ("statistical_assessment", "TEXT DEFAULT NULL"),
        ("model_assessment", "TEXT DEFAULT NULL"),
        ("ensemble_decision", "TEXT DEFAULT NULL"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass

    # Seed Exercises from ExerciseRegistry
    try:
        from exercise_registry import exercise_registry
        for strategy in exercise_registry.list_all():
            cursor.execute("SELECT COUNT(*) FROM exercises WHERE id = ?", (strategy.id,))
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                INSERT INTO exercises (id, name, target_joint, description, target_rom_min, target_rom_max, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy.id,
                    strategy.name,
                    strategy.target_joint,
                    strategy.description,
                    strategy.target_rom_min,
                    strategy.target_rom_max,
                    datetime.now().isoformat()
                ))
    except ImportError:
        # Fallback if circular import during early boot
        cursor.execute("SELECT COUNT(*) FROM exercises")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
            INSERT INTO exercises (id, name, target_joint, description, target_rom_min, target_rom_max, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                "standing-right-arm-abduction",
                "Standing Right-Arm Shoulder Abduction",
                "Right Shoulder & Elbow",
                "Coronal plane shoulder elevation from 0° resting arm up to 90°-100° with extended elbow (~170°-180°), controlled eccentric descent.",
                80.0,
                105.0,
                datetime.now().isoformat()
            ))

    # Seed Standard Baseline Reference if missing
    cursor.execute("SELECT COUNT(*) FROM references_table WHERE id = 'ref-gold-standard-demo'")
    if cursor.fetchone()[0] == 0:
        import numpy as np
        t = np.linspace(0, np.pi, 64)
        elevation = 15.0 + 80.0 * np.sin(t)  # rises from 15 to 95 and returns to 15
        elbow = 172.0 - 5.0 * np.sin(t)      # maintains extended arm (167 - 172 deg)
        timestamps = [round(float(i * 0.05), 3) for i in range(64)] # 3.2s rep
        
        ref_angles = [
            {
                "timestamp": timestamps[i],
                "elevation": round(float(elevation[i]), 2),
                "elbow": round(float(elbow[i]), 2),
                "confidence": 0.96
            }
            for i in range(64)
        ]
        
        cursor.execute("""
        INSERT INTO references_table (
            id, exercise_id, exercise_name, notes, duration_seconds, frame_count,
            joint_angles, landmarks_summary, created_at, is_demo,
            reference_type, version, provenance, patient_id, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "ref-gold-standard-demo",
            "standing-right-arm-abduction",
            "Standing Right-Arm Shoulder Abduction",
            "Clinical baseline standard: Controlled 95° abduction with stabilized trunk and extended elbow (~170°).",
            3.2,
            64,
            json.dumps(ref_angles),
            json.dumps({"avg_confidence": 0.96, "peak_elevation": 95.0, "trunk_stability": "Optimal"}),
            datetime.now().isoformat(),
            1,
            "standard",
            "1.0",
            "Physiotherapy Clinical Protocol v1.0 (Hospital Baseline)",
            None,
            1
        ))

    # Seed Patient-Specific Reference if missing
    cursor.execute("SELECT COUNT(*) FROM references_table WHERE id = 'ref-patient-101-approved'")
    if cursor.fetchone()[0] == 0:
        import numpy as np
        t = np.linspace(0, np.pi, 64)
        # Tailored curve: adapted to 85° max elevation for rotator cuff recovery
        elevation = 15.0 + 70.0 * np.sin(t)
        elbow = 170.0 - 6.0 * np.sin(t)
        timestamps = [round(float(i * 0.05), 3) for i in range(64)]
        
        patient_angles = [
            {
                "timestamp": timestamps[i],
                "elevation": round(float(elevation[i]), 2),
                "elbow": round(float(elbow[i]), 2),
                "confidence": 0.95
            }
            for i in range(64)
        ]

        cursor.execute("""
        INSERT INTO references_table (
            id, exercise_id, exercise_name, notes, duration_seconds, frame_count,
            joint_angles, landmarks_summary, created_at, is_demo,
            reference_type, version, provenance, patient_id, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "ref-patient-101-approved",
            "standing-right-arm-abduction",
            "Standing Right-Arm Shoulder Abduction",
            "Clinician-approved tailored reference for patient-101 (85° target elevation; scapular accommodation).",
            3.2,
            64,
            json.dumps(patient_angles),
            json.dumps({"avg_confidence": 0.95, "peak_elevation": 85.0, "patient_target": "Subacromial Decompression Phase 2"}),
            "2026-10-24T11:00:00",
            1,
            "patient_specific",
            "1.1",
            "Approved by Dr. Sarah Jenkins, PT on 2026-10-24",
            "patient-101",
            1
        ))

    # Seed Initial Demo Sessions if empty
    cursor.execute("SELECT COUNT(*) FROM sessions")
    if cursor.fetchone()[0] == 0:
        # Demo session 1: Good performance
        cursor.execute("""
        INSERT INTO sessions (
            id, timestamp, exercise_id, exercise_name, reference_id,
            repetitions_completed, duration_seconds, angle_stats, comparison_score,
            joint_deviations, model_status, lstm_prediction, warnings, notes, is_demo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "sess-demo-001",
            "2026-10-23T10:15:00",
            "standing-right-arm-abduction",
            "Standing Right-Arm Shoulder Abduction",
            "ref-gold-standard-demo",
            5,
            18.5,
            json.dumps({
                "peak_elevation": 92.4,
                "mean_elevation": 51.2,
                "min_elevation": 12.0,
                "mean_elbow": 169.5,
                "rom": 80.4
            }),
            88.5,
            json.dumps({
                "mean_elevation_deviation": 5.4,
                "max_elevation_deviation": 11.2,
                "mean_elbow_deviation": 4.1,
                "rmse": 6.2
            }),
            "demo_mocap_evaluated",
            json.dumps({
                "source": "REHAB24-6 mocap test repetition",
                "predicted_class": 1,
                "class_label": "correct",
                "probabilities": {"incorrect": 0.12, "correct": 0.88},
                "accuracy_context": "Preliminary held-out participant accuracy: 82.5% (Educational/Research Only)"
            }),
            json.dumps([]),
            "Demo record: High trunk stability and smooth return phase.",
            1
        ))

        # Demo session 2: Slightly restricted ROM
        cursor.execute("""
        INSERT INTO sessions (
            id, timestamp, exercise_id, exercise_name, reference_id,
            repetitions_completed, duration_seconds, angle_stats, comparison_score,
            joint_deviations, model_status, lstm_prediction, warnings, notes, is_demo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "sess-demo-002",
            "2026-10-24T14:30:00",
            "standing-right-arm-abduction",
            "Standing Right-Arm Shoulder Abduction",
            "ref-gold-standard-demo",
            4,
            16.0,
            json.dumps({
                "peak_elevation": 74.8,
                "mean_elevation": 44.0,
                "min_elevation": 14.5,
                "mean_elbow": 158.2,
                "rom": 60.3
            }),
            74.0,
            json.dumps({
                "mean_elevation_deviation": 14.8,
                "max_elevation_deviation": 22.1,
                "mean_elbow_deviation": 13.5,
                "rmse": 16.4
            }),
            "input_requires_validation",
            None,
            json.dumps(["Peak elevation 74.8° below recommended 80°-100° range", "Moderate elbow flexion detected"]),
            "Demo record: User showed minor compensations near apex of movement.",
            1
        ))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_FILE)
