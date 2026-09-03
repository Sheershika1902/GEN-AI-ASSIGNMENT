"""
test_system.py - Automated Verification Suite for IndustroSense AI
Tests RAG pipeline, FAISS similarity search, AI Agent routing, and Audit Logger.
"""

import sys
import io

# Ensure UTF-8 output on Windows consoles
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from documents import get_all_documents, get_document_chunks
from rag import IndustrialRAGPipeline
from agent import IndustrialAIAgent
from logger import log_interaction, get_all_audit_logs


def run_tests():
    print("=" * 60)
    print("RUNNING INDUSTROSENSE AI VERIFICATION SUITE")
    print("=" * 60)

    # 1. Test Documents
    print("\n[TEST 1] Testing Industrial Documents...")
    docs = get_all_documents()
    chunks = get_document_chunks()
    assert len(docs) >= 6, f"Expected at least 6 documents, got {len(docs)}"
    assert len(chunks) >= 6, f"Expected at least 6 chunks, got {len(chunks)}"
    doc_titles = [d["title"] for d in docs]
    print(f"[OK] Found {len(docs)} documents: {', '.join(doc_titles)}")

    # 2. Test RAG & FAISS
    print("\n[TEST 2] Testing RAG Pipeline & FAISS Vector Search...")
    rag = IndustrialRAGPipeline(model_name="all-MiniLM-L6-v2", similarity_threshold=0.35)
    assert rag.is_initialized, f"RAG Pipeline failed to initialize: {rag.initialization_error}"
    print("[OK] SentenceTransformer and FAISS index initialized successfully.")

    test_queries = [
        ("Why is the motor overheating?", "Electric Motor"),
        ("Why is hydraulic pressure decreasing?", "Hydraulic Pump"),
        ("Why is the conveyor belt vibrating?", "Conveyor Belt"),
        ("The air compressor is losing pressure.", "Air Compressor")
    ]

    for query, expected_eq in test_queries:
        res = rag.search(query, top_k=3)
        assert len(res["results"]) == 3, f"Expected 3 results for query '{query}', got {len(res['results'])}"
        top_doc = res["results"][0]
        top_score = top_doc["similarity_score"]
        print(f"[OK] Query: '{query}' -> Top Match: [{top_doc['id']}] {top_doc['title']} (Match: {round(top_score*100, 1)}%)")
        assert not res["is_low_evidence"], f"Expected valid evidence for query '{query}'"

    # Test Low Evidence Handling
    low_ev_query = "What is the capital of France?"
    low_res = rag.search(low_ev_query, top_k=3)
    print(f"[OK] Low Evidence Test: '{low_ev_query}' -> Score: {round(low_res['top_score']*100, 1)}%, Low Evidence: {low_res['is_low_evidence']}")
    assert low_res["is_low_evidence"] or low_res["top_score"] < 0.35, "Expected low evidence flag for irrelevant query"

    guidance = rag.generate_diagnostic_guidance(
        query="Why is the motor overheating?",
        multimodal_context="Problem: Motor overheating | Observations: Wear",
        retrieved_data=rag.search("Why is the motor overheating?", top_k=3),
        image_observations=["Wear"]
    )
    assert guidance["status"] == "success"
    assert len(guidance["possible_causes"]) > 0
    assert len(guidance["recommended_actions"]) > 0
    print(f"[OK] Grounded Diagnostic Guidance generated with {len(guidance['possible_causes'])} causes and {len(guidance['recommended_actions'])} actions.")

    # 3. Test AI Agent Routing
    print("\n[TEST 3] Testing Rule-Based AI Agent Routing...")
    agent = IndustrialAIAgent()

    # Route 1: Retrieval
    r1 = agent.route_query("Why is the motor overheating?")
    assert r1["decision"] == "Retrieval Tool", f"Expected Retrieval Tool, got {r1['decision']}"
    print(f"[OK] Diagnostic query routed to: {r1['decision']}")

    # Route 2: Calculator
    r2 = agent.route_query("Calculate maintenance remaining hours for 5000 interval and 4200 current")
    assert r2["decision"] == "Maintenance Calculator Tool", f"Expected Maintenance Calculator Tool, got {r2['decision']}"
    print(f"[OK] Calculator query routed to: {r2['decision']}")

    # Route 3: Clarification
    r3 = agent.route_query("help")
    assert r3["decision"] == "Clarification Tool", f"Expected Clarification Tool, got {r3['decision']}"
    print(f"[OK] Vague query routed to: {r3['decision']}")

    # Test Calculator Logic
    calc_res = agent.calculate_maintenance_hours(5000.0, 4200.0)
    assert calc_res["valid"] is True
    assert calc_res["remaining_hours"] == 800.0
    print(f"[OK] Calculator test: 5000 - 4200 = {calc_res['remaining_hours']} hrs ({calc_res['status']})")

    # 4. Test Audit Logger
    print("\n[TEST 4] Testing Security & Audit Logger...")
    logged = log_interaction(
        user_query="Why is the motor overheating?",
        image_observations=["Wear"],
        audio_uploaded=False,
        agent_decision="Retrieval Tool",
        retrieved_sources=["DOC-001 - Electric Motor Maintenance Manual"],
        confidence_score=0.85
    )
    assert logged is True
    logs = get_all_audit_logs()
    assert len(logs) > 0
    print(f"[OK] Logged interaction successfully. Total audit log records: {len(logs)}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! PROJECT IS 100% OPERATIONAL.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
