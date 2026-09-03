"""
logger.py - Security & Governance Audit Logging
Maintains an immutable, structured audit trail of diagnostic interactions
without storing sensitive media contents.
"""

import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

AUDIT_LOG_CSV = "audit_log.csv"
AUDIT_LOG_TXT = "audit_log.txt"

FIELDNAMES = [
    "timestamp",
    "user_query",
    "image_observations",
    "audio_uploaded",
    "agent_decision",
    "retrieved_sources",
    "top_confidence_percent"
]


def _ensure_log_files():
    """Initializes audit log files with header if they do not exist."""
    if not os.path.exists(AUDIT_LOG_CSV):
        try:
            with open(AUDIT_LOG_CSV, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
        except Exception:
            pass

    if not os.path.exists(AUDIT_LOG_TXT):
        try:
            with open(AUDIT_LOG_TXT, mode="w", encoding="utf-8") as f:
                f.write("=== INDUSTROSENSE AI - SYSTEM AUDIT LOG ===\n")
                f.write("Format: [TIMESTAMP] | DECISION | QUERY | OBSERVATIONS | AUDIO | SOURCES | CONFIDENCE\n\n")
        except Exception:
            pass


def log_interaction(
    user_query: str,
    image_observations: Optional[List[str]] = None,
    audio_uploaded: bool = False,
    agent_decision: str = "Retrieval Tool",
    retrieved_sources: Optional[List[str]] = None,
    confidence_score: float = 0.0
) -> bool:
    """
    Logs an interaction event to both CSV and TXT audit logs.
    Ensures zero sensitive raw media storage.
    """
    _ensure_log_files()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    obs_str = ", ".join(image_observations) if image_observations else "None"
    sources_str = ", ".join(retrieved_sources) if retrieved_sources else "None"
    sanitized_query = user_query.replace("\n", " ").strip()[:200]  # Truncate length for audit safety

    # 1. Write to CSV Log
    try:
        with open(AUDIT_LOG_CSV, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow({
                "timestamp": timestamp,
                "user_query": sanitized_query,
                "image_observations": obs_str,
                "audio_uploaded": "Yes" if audio_uploaded else "No",
                "agent_decision": agent_decision,
                "retrieved_sources": sources_str,
                "top_confidence_percent": f"{round(confidence_score * 100, 1)}%"
            })
    except Exception as e:
        print(f"Error writing to audit_log.csv: {e}")

    # 2. Write to Human-Readable TXT Log
    try:
        with open(AUDIT_LOG_TXT, mode="a", encoding="utf-8") as f:
            f.write(
                f"[{timestamp}] | DECISION: {agent_decision} | QUERY: \"{sanitized_query}\" | "
                f"OBS: [{obs_str}] | AUDIO: {'Yes' if audio_uploaded else 'No'} | "
                f"SOURCES: [{sources_str}] | CONFIDENCE: {round(confidence_score * 100, 1)}%\n"
            )
    except Exception as e:
        print(f"Error writing to audit_log.txt: {e}")

    return True


def get_all_audit_logs() -> List[Dict[str, str]]:
    """Reads all recorded audit entries from the CSV file."""
    _ensure_log_files()
    records = []
    if os.path.exists(AUDIT_LOG_CSV):
        try:
            with open(AUDIT_LOG_CSV, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
        except Exception as e:
            print(f"Error reading audit_log.csv: {e}")
    return records
