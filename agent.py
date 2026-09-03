"""
agent.py - Rule-Based AI Agent for Tool Selection & Maintenance Calculations
Determines whether to trigger the Retrieval Tool, Maintenance Calculator Tool,
or Clarification Tool, providing a transparent step-by-step decision trace.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class IndustrialAIAgent:
    """
    Rule-Based AI Agent for Industrial Decision Routing.
    Evaluates multimodal query text and intent to select the optimal tool:
    1. Clarification Tool - If input is vague, too short, or lacks context
    2. Maintenance Calculator Tool - For numerical hours/interval calculations
    3. Retrieval Tool - For technical diagnostic & troubleshooting questions
    """

    def __init__(self):
        # Keywords indicating calculation intent
        self.calc_keywords = [
            "calculate", "calculator", "interval", "operating hours", "remaining hours",
            "hours remaining", "maintenance interval", "schedule calculation", "service due",
            "hours left", "hour calculation", "subtract", "formula"
        ]

        # Vague keywords requiring clarification
        self.vague_queries = [
            "help", "broken", "not working", "problem", "issue", "error", "machine",
            "check", "fix", "repair", "stop", "failed", "bad", "motor", "pump"
        ]

    def route_query(
        self,
        query: str,
        image_uploaded: bool = False,
        audio_uploaded: bool = False,
        voice_transcript: str = ""
    ) -> Dict[str, Any]:
        """
        Analyzes query and multimodal inputs, selects tool, and generates decision trace.
        """
        query_clean = query.strip()
        decision_trace: List[str] = []

        # Step 1: Input Length & Modality Check
        decision_trace.append(f"Step 1 [Input Ingestion]: Ingested user query of {len(query_clean)} characters.")
        if image_uploaded:
            decision_trace.append("Step 1b [Modality Check]: Image attachment detected in context.")
        if audio_uploaded or (voice_transcript and len(voice_transcript.strip()) > 0):
            decision_trace.append("Step 1c [Modality Check]: Audio / voice transcript present.")

        # Step 2: Evaluate Clarification Condition (Empty or excessively brief / vague)
        # If user provided no text, or text < 4 chars, and no multimodal inputs
        if len(query_clean) < 4 and not image_uploaded and not (voice_transcript and len(voice_transcript.strip()) > 0):
            decision_trace.append("Step 2 [Validation]: Query length is under minimum threshold (4 chars) with no auxiliary modalities.")
            decision_trace.append("Step 3 [Routing]: Selected 'Clarification Tool' to solicit specific equipment failure details.")
            return {
                "decision": "Clarification Tool",
                "tool_name": "Clarification Tool",
                "status": "clarification_needed",
                "decision_trace": decision_trace,
                "message": (
                    "Your query is too brief or empty. Please describe the specific equipment problem "
                    "(e.g., 'Why is the electric motor overheating?', 'Why is the conveyor belt vibrating?')."
                ),
                "is_executable_query": False
            }

        # Check for vague one-word queries with no context
        if query_clean.lower() in self.vague_queries and not image_uploaded and not voice_transcript:
            decision_trace.append(f"Step 2 [Intent Ambiguity]: Query matches generic non-specific term '{query_clean.lower()}'.")
            decision_trace.append("Step 3 [Routing]: Selected 'Clarification Tool' to request equipment name and operational symptoms.")
            return {
                "decision": "Clarification Tool",
                "tool_name": "Clarification Tool",
                "status": "clarification_needed",
                "decision_trace": decision_trace,
                "message": (
                    f"The query '{query_clean}' is too broad. Please provide the equipment name and symptom "
                    "(e.g., 'Hydraulic pump losing pressure', 'Conveyor belt excessive vibration')."
                ),
                "is_executable_query": False
            }

        # Step 3: Evaluate Maintenance Calculator Condition
        # Detect if query explicitly asks for calculation or interval arithmetic
        calc_match = any(kw in query_clean.lower() for kw in self.calc_keywords)
        
        # Check if query contains numerical interval extraction pattern like "5000 hours interval and 4200 current"
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query_clean)
        has_calc_intent = calc_match or (len(numbers) >= 2 and any(term in query_clean.lower() for term in ["hour", "hrs", "interval"]))

        if has_calc_intent:
            decision_trace.append("Step 2 [Intent Recognition]: Detected numerical / maintenance interval calculation keywords.")
            decision_trace.append("Step 3 [Routing]: Selected 'Maintenance Calculator Tool' for deterministic schedule computation.")
            
            extracted_interval = float(numbers[0]) if len(numbers) >= 1 else 5000.0
            extracted_current = float(numbers[1]) if len(numbers) >= 2 else 4200.0

            return {
                "decision": "Maintenance Calculator Tool",
                "tool_name": "Maintenance Calculator",
                "status": "calculator_ready",
                "decision_trace": decision_trace,
                "suggested_interval": extracted_interval,
                "suggested_current": extracted_current,
                "is_executable_query": True
            }

        # Step 4: Default Route to Retrieval Tool (Diagnostic RAG)
        decision_trace.append("Step 2 [Diagnostic Recognition]: Technical equipment problem identified. Semantic knowledge retrieval required.")
        decision_trace.append("Step 3 [Routing]: Selected 'Retrieval Tool' to query FAISS vector database and retrieve standard operating procedures.")

        return {
            "decision": "Retrieval Tool",
            "tool_name": "Retrieval Tool",
            "status": "retrieval_ready",
            "decision_trace": decision_trace,
            "is_executable_query": True
        }

    @staticmethod
    def calculate_maintenance_hours(
        interval_hours: float,
        current_hours: float
    ) -> Dict[str, Any]:
        """
        Executes deterministic maintenance interval calculations:
        Remaining Hours = Maintenance Interval - Current Operating Hours
        """
        # Input validation
        if interval_hours <= 0:
            return {
                "error": "Maintenance interval must be a positive number greater than 0.",
                "valid": False
            }
        
        if current_hours < 0:
            return {
                "error": "Current operating hours cannot be negative.",
                "valid": False
            }

        remaining_hours = interval_hours - current_hours
        percent_used = min(100.0, max(0.0, (current_hours / interval_hours) * 100.0))
        
        # Determine operational maintenance status
        if remaining_hours < 0:
            status = "CRITICAL OVERDUE"
            status_color = "red"
            recommendation = "Equipment has exceeded its service interval. Schedule emergency maintenance immediately."
        elif remaining_hours <= (interval_hours * 0.15):  # Within last 15% of interval
            status = "MAINTENANCE DUE SOON"
            status_color = "orange"
            recommendation = "Service threshold approaching. Prepare parts, lubricants, and technician work orders."
        else:
            status = "OPERATIONAL NORMAL"
            status_color = "green"
            recommendation = "Operating within normal maintenance window. Continue routine telemetry monitoring."

        return {
            "valid": True,
            "interval_hours": interval_hours,
            "current_hours": current_hours,
            "remaining_hours": round(remaining_hours, 2),
            "percent_used": round(percent_used, 1),
            "status": status,
            "status_color": status_color,
            "recommendation": recommendation,
            "formula": "Remaining Hours = Maintenance Interval (hrs) - Current Operating Hours (hrs)"
        }
