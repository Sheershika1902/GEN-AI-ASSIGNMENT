# 🏭 IndustroSense AI
### A Multimodal Responsible Generative AI Assistant for Industrial Equipment Diagnostics and Documentation

---

## 📌 Project Overview

**IndustroSense AI** is an academic-grade, multimodal, responsible AI assistant built with **Streamlit**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, and **FAISS (Facebook AI Similarity Search)**. It assists maintenance technicians and plant reliability engineers in diagnosing industrial machinery issues across electric motors, hydraulic pumps, conveyor belts, and air compressors.

### 🌟 Key Architecture & Features:
1. **RAG Pipeline (Retrieval-Augmented Generation):**
   - Industrial Standard Operating Procedures (SOPs) & Incident Logs
   - Dense semantic embeddings generated with `all-MiniLM-L6-v2`
   - FAISS vector indexing (`IndexFlatIP` on L2-normalized embeddings for exact Cosine Similarity)
   - Top-3 retrieval with transparency scores and grounded diagnostic synthesis
2. **Deterministic AI Agent Router:**
   - Evaluates input structure and intent
   - Dynamically selects **Retrieval Tool**, **Maintenance Calculator Tool**, or **Clarification Tool**
   - Displays real-time decision reasoning traces
3. **Multimodal Late Fusion:**
   - Text Problem Descriptions + Visual Observation Tags (Corrosion, Leakage, Crack, Wear, etc.) + Voice-Note Transcripts
   - Transparent non-simulated local observation verification
4. **Responsible AI & Governance:**
   - Complete source citations for every retrieved fact
   - Semantic similarity calibration vs physical diagnostic certainty
   - Automated **Low Evidence Alerts** when queries fall below confidence thresholds
   - Safety warnings emphasizing Lockout/Tagout (LOTO) and qualified human oversight
   - Immutable security and audit logging in `audit_log.csv` and `audit_log.txt` (zero raw media storage)
   - Interactive System **Model Card**

---

## 📂 Project Structure

```
IndustroSense_AI/
│
├── app.py              # Main Streamlit web application & UI
├── documents.py        # Industrial knowledge base documents & chunking
├── rag.py              # Embeddings, FAISS vector database & grounded synthesis
├── agent.py            # Rule-based AI agent router & Maintenance Calculator
├── logger.py           # Security & audit logging system
├── requirements.txt    # Minimal project dependencies
└── README.md           # Setup instructions & documentation
```

---

## 🚀 Quickstart Guide (VS Code on Windows)

Follow these step-by-step instructions to run IndustroSense AI on Windows:

### Step 1: Open the Project in VS Code
1. Launch **Visual Studio Code**.
2. Click **File > Open Folder...** and select the `Assignment Gen ai` (or `IndustroSense_AI`) folder.

### Step 2: Open the Integrated Terminal
- Press ``Ctrl + ` `` (Backtick) or go to **Terminal > New Terminal** in the top menu.

### Step 3: Create a Virtual Environment (Recommended)
In the terminal, run:
```powershell
python -m venv venv
```

### Step 4: Activate the Virtual Environment
- **PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If you encounter an execution policy restriction, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` and try activating again)*
- **Command Prompt (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```

### Step 5: Install Dependencies
Run:
```powershell
pip install -r requirements.txt
```

### Step 6: Launch the Application
Run:
```powershell
streamlit run app.py
```
*The Streamlit web interface will automatically open in your default browser at `http://localhost:8501`.*

---

## 🧪 Testing Checklist & Sample Queries

Try these sample queries in the **Multimodal Diagnostics** tab to test all system features:

| Test Case | Sample Query / Action | Expected Agent Decision & Behavior |
|---|---|---|
| **1. Motor Overheating** | `"Why is the motor overheating?"` | **Retrieval Tool** → Retrieves `DOC-001` (Electric Motor Manual) & `DOC-002` (Bearing Guide). |
| **2. Hydraulic Pressure** | `"Why is hydraulic pressure decreasing?"` | **Retrieval Tool** → Retrieves `DOC-003` (Hydraulic Pump SOP) with causes like seal leaks & cavitation. |
| **3. Conveyor Vibration** | `"Why is the conveyor belt vibrating?"` | **Retrieval Tool** → Retrieves `DOC-004` (Conveyor Belt Manual) with alignment and roller checks. |
| **4. Air Compressor** | `"The air compressor is losing pressure."` | **Retrieval Tool** → Retrieves `DOC-005` (Air Compressor Guide) highlighting leaks and filter blockages. |
| **5. Multimodal Image Fusion** | Query: `"Motor running hot"` + Select observation: `Wear` | Fuses text + visual tags; prioritizes physical bearing wear checks in guidance. |
| **6. Maintenance Calculator** | `"Calculate maintenance remaining for 5000 interval and 4200 current"` | **Maintenance Calculator Tool** → Computes `5000 - 4200 = 800 hours remaining (16% remaining)`. |
| **7. Clarification Request** | `"help"` or `"broken"` (very short/vague) | **Clarification Tool** → Prompts user for specific machinery symptoms without crashing. |
| **8. Low Evidence Handling** | `"What is the best recipe for chocolate cake?"` | Flags **Low Evidence Alert** (confidence < 35%) and advises qualified technical consultation. |

---

## 🛠️ Troubleshooting Guide

### 1. `python` or `pip` is not recognized
- Ensure Python 3.9+ is installed from [python.org](https://www.python.org/downloads/).
- During installation, make sure to check **"Add Python to PATH"**.
- Restart VS Code after installing Python.

### 2. PowerShell Script Execution Error (`Activate.ps1 cannot be loaded`)
Run this command in your VS Code PowerShell terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then run `.\venv\Scripts\Activate.ps1` again.

### 3. `streamlit` is not recognized
Ensure your virtual environment is active (you should see `(venv)` at the beginning of the terminal line), or run:
```powershell
python -m streamlit run app.py
```

### 4. FAISS Installation on Windows
If `faiss-cpu` fails to build, make sure you installed `faiss-cpu` (not `faiss`), which provides pre-compiled Windows wheels:
```powershell
pip install faiss-cpu
```

---

## 🛡️ Responsible AI & Safety Notice

> **IMPORTANT:** IndustroSense AI is intended strictly as an educational and decision-support tool. It does not replace certified mechanical/electrical reliability engineers or standard industrial Lockout/Tagout (LOTO) protocols.
