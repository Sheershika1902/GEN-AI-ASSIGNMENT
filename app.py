"""
app.py - IndustroSense AI
A Multimodal Responsible Generative AI Assistant for Industrial Equipment Diagnostics
Streamlit Web Application
"""

import os
import streamlit as st
from PIL import Image
from typing import List, Dict, Any

from documents import get_all_documents
from rag import IndustrialRAGPipeline
from agent import IndustrialAIAgent
from logger import log_interaction, get_all_audit_logs

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="IndustroSense AI - Industrial Diagnostics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Industrial Aesthetics
st.markdown("""
<style>
    /* Main container and typography styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        color: #f8fafc;
    }
    .main-header h1 {
        color: #38bdf8;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #94a3b8;
        margin-top: 6px;
        font-size: 1.05rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-blue { background-color: #0284c7; color: white; }
    .badge-green { background-color: #16a34a; color: white; }
    .badge-amber { background-color: #d97706; color: white; }
    .badge-red { background-color: #dc2626; color: white; }
    
    .card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .card-dark {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        color: #f1f5f9;
        margin-bottom: 16px;
    }
    .trace-box {
        background-color: #1e293b;
        color: #38bdf8;
        font-family: 'Courier New', monospace;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        border-left: 4px solid #38bdf8;
        margin-bottom: 16px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .safety-alert {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        color: #92400e;
        padding: 14px;
        border-radius: 6px;
        margin-top: 18px;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Singleton RAG Pipeline & Agent Initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing RAG Pipeline & Embedding Model...")
def load_rag_pipeline() -> IndustrialRAGPipeline:
    """Initializes and caches the SentenceTransformers + FAISS RAG pipeline."""
    return IndustrialRAGPipeline(model_name="all-MiniLM-L6-v2", similarity_threshold=0.35)


@st.cache_resource
def load_agent() -> IndustrialAIAgent:
    """Initializes the rule-based AI agent."""
    return IndustrialAIAgent()


# Initialize core components
rag_pipeline = load_rag_pipeline()
agent = load_agent()

# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/engineering.png", width=60)
    st.title("IndustroSense AI")
    st.caption("Responsible Multimodal Industrial Diagnostic System")
    st.markdown("---")

    selected_view = st.radio(
        "Navigation",
        [
            "🔍 Multimodal Diagnostics",
            "⏱️ Maintenance Calculator",
            "📚 Knowledge Base Explorer",
            "📋 Security & Audit Logs",
            "🛡️ Model Card & Governance"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("⚙️ System Configuration")
    sim_threshold = st.slider(
        "Retrieval Confidence Threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="Queries with top similarity below this value trigger Low Evidence warnings."
    )
    rag_pipeline.similarity_threshold = sim_threshold

    st.markdown("---")
    st.subheader("⚡ Quick Sample Queries")
    sample_queries = [
        "Why is the motor overheating?",
        "Why is hydraulic pressure decreasing?",
        "Why is the conveyor belt vibrating?",
        "The air compressor is losing pressure.",
        "Calculate maintenance remaining for 5000 interval and 4200 current"
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq[:15]}"):
            st.session_state["query_input"] = sq

    st.markdown("---")
    st.caption("Developed for Academic Demonstrations • Grounded RAG + FAISS")


# ---------------------------------------------------------
# VIEW 1: MULTIMODAL DIAGNOSTICS (MAIN APPLICATION)
# ---------------------------------------------------------
if selected_view == "🔍 Multimodal Diagnostics":
    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1>🏭 IndustroSense AI</h1>
        <p>A Multimodal Responsible Generative AI Assistant for Industrial Equipment Diagnostics and Documentation</p>
        <span class="status-badge badge-blue">RAG Pipeline: FAISS + all-MiniLM-L6-v2</span>
        <span class="status-badge badge-green">AI Agent: Active</span>
        <span class="status-badge badge-amber">Fusion: Late Multimodal</span>
    </div>
    """, unsafe_allow_html=True)

    # 1. INPUT SECTION
    st.subheader("1. Industrial Equipment Input & Multimodal Ingestion")
    
    col_text, col_media = st.columns([1.2, 1])

    with col_text:
        default_query = st.session_state.get("query_input", "")
        user_query = st.text_area(
            "Equipment Problem Description (Text Query):",
            value=default_query,
            placeholder="e.g., Why is the motor overheating? Or: Hydraulic pressure is dropping rapidly.",
            height=130,
            help="Enter the operational symptoms, abnormal sounds, or equipment failure details."
        )

    with col_media:
        st.markdown("**Optional Multimodal Inputs:**")
        
        tab_img, tab_audio = st.tabs(["📷 Image Ingestion", "🎙️ Audio / Voice Ingestion"])
        
        # Image Modality Tab
        uploaded_image = None
        selected_observations = []
        with tab_img:
            uploaded_image_file = st.file_uploader(
                "Upload Equipment Photo (JPG, JPEG, PNG):",
                type=["jpg", "jpeg", "png"],
                help="Upload an on-site photo of the machinery or component."
            )
            if uploaded_image_file is not None:
                try:
                    uploaded_image = Image.open(uploaded_image_file)
                    st.image(uploaded_image, caption=f"Uploaded: {uploaded_image_file.name}", width=260)
                except Exception as e:
                    st.error(f"Error opening image: {e}")

            st.markdown("**Visual Observation Checklist:**")
            st.caption("🔬 *Local / User-Confirmed Observation Tagging (Transparent Non-Simulated Analysis)*")
            obs_options = [
                "Corrosion",
                "Leakage",
                "Crack",
                "Wear",
                "Damaged Component",
                "No Visible Damage"
            ]
            selected_observations = st.multiselect(
                "Select visible physical anomalies:",
                obs_options,
                default=[] if not uploaded_image else ["Wear"]
            )

        # Audio Modality Tab
        uploaded_audio_file = None
        voice_transcript = ""
        with tab_audio:
            uploaded_audio_file = st.file_uploader(
                "Upload Field Audio Note (MP3, WAV):",
                type=["mp3", "wav"],
                help="Upload recorded acoustic noise or technician voice note."
            )
            if uploaded_audio_file is not None:
                st.audio(uploaded_audio_file)

            voice_transcript = st.text_area(
                "Voice-Note Transcript (Manual Field Transcript):",
                placeholder="e.g., Operator noted heavy grinding sound near bearing housing during night shift.",
                height=70,
                help="Manual transcript of field voice notes or acoustic observations."
            )

    # Action Button
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze Problem", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Clear Inputs"):
            st.session_state["query_input"] = ""
            st.rerun()

    # ---------------------------------------------------------
    # MULTIMODAL LATE FUSION & AGENT EXECUTION
    # ---------------------------------------------------------
    if analyze_clicked or user_query.strip():
        st.markdown("---")
        st.subheader("2. Diagnostics & Responsible AI Execution Results")

        # Step A: Multimodal Late Fusion
        has_image = uploaded_image_file is not None
        has_audio = uploaded_audio_file is not None
        has_transcript = bool(voice_transcript.strip())
        has_obs = bool(selected_observations)

        # Build late fusion context components
        fusion_components = []
        if user_query.strip():
            fusion_components.append(f"Problem Description: {user_query.strip()}")
        if has_obs:
            fusion_components.append(f"Visual Observations: {', '.join(selected_observations)}")
        if has_transcript:
            fusion_components.append(f"Voice Transcript: {voice_transcript.strip()}")

        if not fusion_components:
            multimodal_diagnostic_context = "No input provided."
        else:
            multimodal_diagnostic_context = " | ".join(fusion_components)

        # Step B: AI Agent Routing
        agent_result = agent.route_query(
            query=user_query,
            image_uploaded=has_image,
            audio_uploaded=has_audio,
            voice_transcript=voice_transcript
        )

        selected_tool = agent_result["decision"]
        decision_trace = agent_result["decision_trace"]

        # -----------------------------------------------------
        # RESULT SECTION ORDER (As strictly required):
        # 1. AI Agent Decision
        # 2. Agent Decision Trace
        # 3. Multimodal Diagnostic Context
        # 4. Retrieved Technical Sources
        # 5. Retrieval / Similarity Scores
        # 6. Diagnostic Guidance
        # 7. Recommended Actions
        # 8. Responsible AI / Safety Notice
        # -----------------------------------------------------

        # 1. AI Agent Decision
        if selected_tool == "Retrieval Tool":
            badge_class = "badge-blue"
        elif selected_tool == "Maintenance Calculator Tool":
            badge_class = "badge-green"
        else:
            badge_class = "badge-amber"

        st.markdown(f"""
        <div style="background-color: #f1f5f9; padding: 14px 20px; border-radius: 8px; border-left: 6px solid #0284c7; margin-bottom: 14px;">
            <h3 style="margin:0; color: #0f172a;">🤖 AI Agent Decision: <span class="status-badge {badge_class}" style="font-size:1rem;">{selected_tool}</span></h3>
        </div>
        """, unsafe_allow_html=True)

        # 2. Agent Decision Trace
        with st.expander("🔍 View AI Agent Decision Trace", expanded=True):
            trace_html = "<br>".join([f"• {step}" for step in decision_trace])
            st.markdown(f"""
            <div class="trace-box">
                {trace_html}
            </div>
            """, unsafe_allow_html=True)

        # 3. Multimodal Diagnostic Context
        st.markdown("#### 🧩 Multimodal Diagnostic Context")
        st.markdown("""
        <div class="card">
            <strong>Combined Late Fusion Context:</strong><br>
            <code>{}</code>
        </div>
        """.format(multimodal_diagnostic_context), unsafe_allow_html=True)

        # Handle Routing Scenarios
        if selected_tool == "Clarification Tool":
            st.warning(f"⚠️ **Clarification Requested:** {agent_result['message']}")
            st.info("💡 **Tip:** Try one of the quick sample queries in the left sidebar to see the RAG retrieval pipeline in action.")
            
            # Log clarification interaction
            log_interaction(
                user_query=user_query,
                image_observations=selected_observations,
                audio_uploaded=has_audio,
                agent_decision="Clarification Tool",
                retrieved_sources=["None"],
                confidence_score=0.0
            )

        elif selected_tool == "Maintenance Calculator Tool":
            st.markdown("#### ⏱️ Maintenance Calculation Execution")
            st.info("The AI Agent routed this query to the Maintenance Calculator Tool.")
            
            s_interval = agent_result.get("suggested_interval", 5000.0)
            s_current = agent_result.get("suggested_current", 4200.0)

            c1, c2 = st.columns(2)
            with c1:
                input_interval = st.number_input("Total Maintenance Interval (Hours):", min_value=1.0, value=float(s_interval), step=100.0)
            with c2:
                input_current = st.number_input("Current Operating Hours:", min_value=0.0, value=float(s_current), step=50.0)

            calc_res = agent.calculate_maintenance_hours(input_interval, input_current)
            
            if calc_res.get("valid"):
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Interval", f"{calc_res['interval_hours']} hrs")
                with col_m2:
                    st.metric("Current Operating", f"{calc_res['current_hours']} hrs")
                with col_m3:
                    rem = calc_res['remaining_hours']
                    delta_color = "normal" if rem > 0 else "inverse"
                    st.metric("Remaining Hours", f"{rem} hrs", delta=f"{calc_res['percent_used']}% elapsed", delta_color=delta_color)

                st.markdown(f"""
                <div class="card">
                    <strong>Formula:</strong> <code>{calc_res['formula']}</code><br>
                    <strong>Calculation:</strong> <code>{calc_res['interval_hours']} - {calc_res['current_hours']} = {calc_res['remaining_hours']} Hours</code><br>
                    <strong>Operational Status:</strong> <span class="status-badge badge-{calc_res['status_color']}">{calc_res['status']}</span><br>
                    <strong>Recommendation:</strong> {calc_res['recommendation']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(calc_res.get("error"))

            # Log calculator interaction
            log_interaction(
                user_query=user_query,
                image_observations=selected_observations,
                audio_uploaded=has_audio,
                agent_decision="Maintenance Calculator Tool",
                retrieved_sources=["Internal Calculator Formula"],
                confidence_score=1.0
            )

        elif selected_tool == "Retrieval Tool":
            # Execute RAG Retrieval
            # Formulate query for retrieval (combining text + observations + transcript)
            retrieval_query = f"{user_query} {' '.join(selected_observations)} {voice_transcript}".strip()
            
            with st.spinner("Searching FAISS vector index & analyzing standard operating procedures..."):
                retrieval_output = rag_pipeline.search(retrieval_query, top_k=3)
                guidance_output = rag_pipeline.generate_diagnostic_guidance(
                    query=user_query,
                    multimodal_context=multimodal_diagnostic_context,
                    retrieved_data=retrieval_output,
                    image_observations=selected_observations
                )

            # 4. Retrieved Technical Sources & 5. Retrieval / Similarity Scores
            st.markdown("#### 📚 Retrieved Technical Sources & Similarity Scores")
            st.caption("Top-3 relevant knowledge base documents retrieved from FAISS vector database (Model: `all-MiniLM-L6-v2`)")

            retrieved_docs = retrieval_output.get("results", [])
            retrieved_source_titles = []

            if not retrieved_docs or retrieval_output.get("is_low_evidence", False):
                st.error("""
                ⚠️ **Low Evidence Alert:** Insufficient relevant evidence was found in the knowledge base (Top score is below confidence threshold).
                Please provide additional details or consult a qualified technician.
                """)
            
            cols = st.columns(len(retrieved_docs) if retrieved_docs else 1)
            for i, doc in enumerate(retrieved_docs):
                retrieved_source_titles.append(f"{doc['id']} - {doc['title']}")
                with cols[i]:
                    score_pct = doc["confidence_percent"]
                    badge_col = "badge-green" if score_pct >= 60 else ("badge-blue" if score_pct >= 35 else "badge-amber")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="status-badge badge-blue">{doc['id']}</span>
                            <span class="status-badge {badge_col}">{score_pct}% Match</span>
                        </div>
                        <h4 style="margin: 8px 0 4px 0; font-size:1.05rem;">{doc['title']}</h4>
                        <p style="color:#64748b; font-size:0.85rem; margin-bottom:8px;"><strong>Category:</strong> {doc['category']}<br><strong>Equipment:</strong> {doc['equipment_type']}</p>
                        <hr style="margin: 6px 0;">
                        <p style="font-size:0.88rem; color:#334155; line-height:1.35;">"{doc['raw_content'][:160]}..."</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Transparency Note on Similarity
            st.info("""
            ℹ️ **Source & Similarity Transparency:** Retrieval scores represent cosine similarity between semantic vector embeddings. 
            Higher similarity indicates topical relevance in technical documentation, not diagnostic certainty.
            """)

            # 6. Diagnostic Guidance & 7. Recommended Actions
            st.markdown("#### 🩺 Grounded Diagnostic Guidance")
            
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid #38bdf8;">
                <p style="font-size:1.02rem; margin-bottom:8px;"><strong>Diagnostic Summary:</strong> {guidance_output['summary']}</p>
            </div>
            """, unsafe_allow_html=True)

            g_col1, g_col2 = st.columns(2)
            
            with g_col1:
                st.markdown("##### 🔍 Possible Causes")
                for cause in guidance_output.get("possible_causes", []):
                    st.markdown(f"• **Possible cause:** {cause}")

            with g_col2:
                st.markdown("##### 🛠️ Recommended Actions")
                for action in guidance_output.get("recommended_actions", []):
                    st.markdown(f"• **Action:** {action}")

            # Evidence Used Breakdown
            with st.expander("📑 View Complete Grounded Evidence Citations"):
                for src in guidance_output.get("evidence_sources", []):
                    st.markdown(f"**[{src['id']}] {src['title']}** (*Match: {src['confidence_percent']}%*)")
                    st.markdown(f"> {src['excerpt']}")
                    st.markdown("---")

            # 8. Responsible AI / Safety Notice
            st.markdown(f"""
            <div class="safety-alert">
                <strong>⚠️ Responsible AI & Industrial Safety Notice:</strong><br>
                This system provides decision-support information and should not replace qualified engineers or technicians for safety-critical industrial decisions.
                Always follow Lockout/Tagout (LOTO) protocols and verify electrical/mechanical systems physically prior to servicing.
            </div>
            """, unsafe_allow_html=True)

            # Log interaction to audit trail
            top_conf = retrieval_output.get("top_score", 0.0)
            log_interaction(
                user_query=user_query,
                image_observations=selected_observations,
                audio_uploaded=has_audio,
                agent_decision="Retrieval Tool",
                retrieved_sources=retrieved_source_titles,
                confidence_score=top_conf
            )


# ---------------------------------------------------------
# VIEW 2: MAINTENANCE CALCULATOR (DEDICATED TOOL VIEW)
# ---------------------------------------------------------
elif selected_view == "⏱️ Maintenance Calculator":
    st.markdown("""
    <div class="main-header">
        <h1>⏱️ Maintenance Interval Calculator</h1>
        <p>Deterministic Tool for Remaining Operating Life and Maintenance Schedule Tracking</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Equipment Service Schedule Calculator")
    st.markdown("Use this tool to determine remaining operational hours and schedule preventive maintenance.")

    c1, c2 = st.columns(2)
    with c1:
        interval_input = st.number_input("Total Maintenance Interval (Hours):", min_value=1.0, value=5000.0, step=100.0, help="Manufacturer recommended interval between major service.")
    with c2:
        current_input = st.number_input("Current Operating Hours:", min_value=0.0, value=4200.0, step=50.0, help="Cumulative machine operating hours from telemetry.")

    calc_data = agent.calculate_maintenance_hours(interval_input, current_input)

    if calc_data.get("valid"):
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Design Service Interval", f"{calc_data['interval_hours']} hrs")
        with m2:
            st.metric("Current Running Time", f"{calc_data['current_hours']} hrs")
        with m3:
            rem = calc_data['remaining_hours']
            st.metric("Remaining Hours", f"{rem} hrs", delta=f"{calc_data['percent_used']}% used", delta_color="normal" if rem > 0 else "inverse")

        st.progress(min(1.0, calc_data['percent_used'] / 100.0))

        st.markdown(f"""
        <div class="card">
            <h4>Calculation Details</h4>
            <p><strong>Standard Formula:</strong> <code>{calc_data['formula']}</code></p>
            <p><strong>Computed Result:</strong> <code>{calc_data['interval_hours']} - {calc_data['current_hours']} = {calc_data['remaining_hours']} Hours</code></p>
            <p><strong>Operational Status:</strong> <span class="status-badge badge-{calc_data['status_color']}">{calc_data['status']}</span></p>
            <p><strong>Recommended Procedure:</strong> {calc_data['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(calc_data.get("error"))


# ---------------------------------------------------------
# VIEW 3: KNOWLEDGE BASE EXPLORER
# ---------------------------------------------------------
elif selected_view == "📚 Knowledge Base Explorer":
    st.markdown("""
    <div class="main-header">
        <h1>📚 Industrial Knowledge Base Explorer</h1>
        <p>Standard Operating Procedures, Technical Manuals, and Incident Histories</p>
    </div>
    """, unsafe_allow_html=True)

    docs = get_all_documents()
    st.markdown(f"**Total Registered Documents:** {len(docs)}")

    for doc in docs:
        with st.expander(f"📖 [{doc['id']}] {doc['title']} ({doc['equipment_type']})", expanded=True):
            st.markdown(f"**Category:** `{doc['category']}` | **Equipment:** `{doc['equipment_type']}`")
            st.markdown(f"**Standard Content:**\n> {doc['content']}")
            
            k1, k2 = st.columns(2)
            with k1:
                st.markdown("**Typical Failure Causes:**")
                for c in doc.get("typical_causes", []):
                    st.markdown(f"- {c}")
            with k2:
                st.markdown("**Recommended Standard Actions:**")
                for a in doc.get("recommended_actions", []):
                    st.markdown(f"- {a}")


# ---------------------------------------------------------
# VIEW 4: SECURITY & AUDIT LOGS
# ---------------------------------------------------------
elif selected_view == "📋 Security & Audit Logs":
    st.markdown("""
    <div class="main-header">
        <h1>📋 Security & Audit Trail</h1>
        <p>Immutable Interaction Logs for Governance, Traceability, and Safety Compliance</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <strong>Privacy & Security Policy:</strong><br>
        • No sensitive raw binary files (images/audio) are stored on disk.<br>
        • All diagnostic queries, agent decisions, and retrieved citations are tracked for regulatory review.<br>
        • Log Files: <code>audit_log.csv</code> and <code>audit_log.txt</code>
    </div>
    """, unsafe_allow_html=True)

    logs = get_all_audit_logs()
    if logs:
        st.markdown(f"**Recorded Diagnostic Sessions:** {len(logs)}")
        st.dataframe(logs, use_container_width=True)
    else:
        st.info("No audit logs recorded yet. Run a diagnostic query in the 'Multimodal Diagnostics' tab to generate logs.")


# ---------------------------------------------------------
# VIEW 5: MODEL CARD & GOVERNANCE
# ---------------------------------------------------------
elif selected_view == "🛡️ Model Card & Governance":
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Model Card & System Governance</h1>
        <p>Transparency, Operational Scope, Limitations, and Responsible AI Architecture</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        ### 🎯 System Purpose
        **IndustroSense AI** is an academic-grade decision-support system designed to assist plant technicians and reliability engineers in troubleshooting industrial machinery failures (motors, pumps, compressors, conveyors).

        ### ⚙️ Core Capabilities
        - **Semantic Knowledge Retrieval (RAG):** Dense vector indexing with `all-MiniLM-L6-v2` and FAISS for sub-millisecond SOP retrieval.
        - **Deterministic AI Routing:** Rule-based agent dynamically dispatches queries to RAG retrieval, maintenance calculators, or clarification prompts.
        - **Multimodal Late Fusion:** Correlates textual symptoms, verified visual inspection tags, and field audio transcripts.
        - **Grounded Response Generation:** Synthesizes diagnostic guidance strictly from retrieved technical manuals with explicit evidence citations.
        """)

    with c2:
        st.markdown("""
        ### ⚠️ Known Limitations
        - **Decision Support Only:** The system is **not** an autonomous control system. It does not replace human engineers.
        - **Knowledge Boundary:** Diagnosis is constrained to the registered standard operating procedures in `documents.py`.
        - **Visual Modality:** Visual observation tags require user verification; no black-box vision model predictions are accepted without review.

        ### 🛡️ Safety & Responsible AI Principles
        - **Source Grounding:** All claims cite specific document IDs.
        - **Uncertainty Calibration:** Non-deterministic claims use phrases such as *"Possible cause"* and *"Based on available evidence"*.
        - **Low Evidence Handling:** Automatically alerts users when query relevance is below threshold.
        - **Audit Traceability:** Every query and decision is logged with timestamps for review.
        """)
