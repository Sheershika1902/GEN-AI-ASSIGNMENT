"""
documents.py - Industrial Equipment Knowledge Base
Contains technical standard operating procedures (SOPs), maintenance manuals,
and incident history for industrial machinery diagnostics.
"""

from typing import List, Dict, Any

# Standard Industrial Knowledge Base Documents
INDUSTRIAL_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "DOC-001",
        "title": "Electric Motor Maintenance Manual",
        "category": "Electrical & Rotating Machinery",
        "equipment_type": "Electric Motor",
        "content": (
            "Electric motors may overheat due to excessive electrical load, poor ventilation, "
            "damaged bearings, or inadequate lubrication. Regular inspection of the cooling system "
            "and bearings is recommended. Ensure that the cooling fan is clean, ambient temperature "
            "is within rated limits, and motor housing is free from dust buildup."
        ),
        "typical_causes": [
            "Excessive electrical overload",
            "Poor ventilation / clogged cooling fan",
            "Damaged or worn bearings",
            "Inadequate or contaminated lubrication"
        ],
        "recommended_actions": [
            "Measure operating current against nameplate full-load amps (FLA)",
            "Inspect and clean cooling fins, fan cover, and ventilation paths",
            "Check bearing temperature and perform acoustic/vibration testing",
            "Replenish or replace lubrication grease according to manufacturer spec"
        ],
        "keywords": ["motor", "overheating", "temperature", "ventilation", "electrical load", "cooling"]
    },
    {
        "id": "DOC-002",
        "title": "Electric Motor Bearing Guide",
        "category": "Mechanical Components",
        "equipment_type": "Electric Motor Bearings",
        "content": (
            "Damaged or poorly lubricated bearings can cause excessive friction, vibration, "
            "grinding sounds, and increased operating temperature. Bearings should be inspected "
            "and lubricated regularly. Use precision vibration analysis to detect early stage flaking, "
            "pitting, or raceway fatigue."
        ),
        "typical_causes": [
            "Lubrication starvation or dried grease",
            "Excessive radial or axial mechanical preload",
            "Bearing raceway fatigue, flaking, or spalling",
            "Contamination by moisture, metal particles, or dirt"
        ],
        "recommended_actions": [
            "Perform spectrum vibration analysis for bearing fault frequencies (BPFO, BPFI)",
            "Check grease color and consistency; flush and re-grease if contaminated",
            "Inspect shaft alignment and dynamic balance",
            "Replace bearing assembly if grinding sounds or physical wear are detected"
        ],
        "keywords": ["bearing", "vibration", "grinding", "friction", "lubrication", "grease", "sound"]
    },
    {
        "id": "DOC-003",
        "title": "Hydraulic Pump SOP",
        "category": "Hydraulic Systems",
        "equipment_type": "Hydraulic Pump & Actuators",
        "content": (
            "Hydraulic pressure loss may occur due to fluid leakage, blocked filters, "
            "air entering the hydraulic system, cavitation, or damaged seals. Inspect hoses, "
            "filters and seals regularly. Verify oil level in the reservoir and confirm pump "
            "relief valve calibration."
        ),
        "typical_causes": [
            "External or internal hydraulic fluid leakage",
            "Clogged suction strainer or return-line filter",
            "Air entrainment or pump suction cavitation",
            "Worn cylinder seals, O-rings, or relief valve malfunction"
        ],
        "recommended_actions": [
            "Conduct visual and dye inspection for hose, fitting, and manifold leaks",
            "Inspect differential pressure indicator on filters and replace elements",
            "Bleed air from the hydraulic circuit and check reservoir fluid levels",
            "Test and calibrate system pressure relief valve setting"
        ],
        "keywords": ["hydraulic", "pressure loss", "leakage", "fluid", "cavitation", "filter", "seal"]
    },
    {
        "id": "DOC-004",
        "title": "Conveyor Belt Maintenance Manual",
        "category": "Material Handling",
        "equipment_type": "Conveyor Belt System",
        "content": (
            "Conveyor belt vibration may occur due to misalignment, damaged rollers, "
            "uneven loading, or loose mechanical components. Regular alignment and roller "
            "inspection are recommended. Check belt tension and ensure drive pulley lagging "
            "is intact."
        ),
        "typical_causes": [
            "Belt tracking misalignment or uneven tension",
            "Seized, eccentric, or damaged idler rollers",
            "Off-center or shock material loading",
            "Loose structural fasteners or pulley mounting bolts"
        ],
        "recommended_actions": [
            "Verify belt tracking and adjust take-up frames for balanced tension",
            "Inspect all idler and return rollers for free rotation and bearing play",
            "Ensure chute discharge feeds material centrally onto the belt",
            "Torque all structural mounting bolts and inspect pulley bearings"
        ],
        "keywords": ["conveyor", "vibration", "misalignment", "roller", "belt", "tension", "pulley"]
    },
    {
        "id": "DOC-005",
        "title": "Air Compressor Maintenance Guide",
        "category": "Pneumatic Systems",
        "equipment_type": "Rotary / Reciprocating Air Compressor",
        "content": (
            "Air compressors may experience reduced pressure due to air leaks, blocked filters, "
            "damaged valves, or excessive wear. Check air filters and connections during "
            "maintenance. Ensure minimum pressure valve operates smoothly and condensation is drained."
        ),
        "typical_causes": [
            "Pneumatic distribution line leaks or loose quick-connect fittings",
            "Heavy particulate fouling on air intake filter cartridges",
            "Faulty unloader valve, intake valve, or minimum pressure check valve",
            "Internal screw element or piston ring wear"
        ],
        "recommended_actions": [
            "Perform ultrasonic leak detection along all pressurized air lines",
            "Clean or replace intake air filter elements",
            "Test intake control / unloader valve actuation",
            "Inspect separator element and drain automatic moisture traps"
        ],
        "keywords": ["air compressor", "pressure", "air leak", "filter", "valve", "pneumatic"]
    },
    {
        "id": "DOC-006",
        "title": "Industrial Incident Log",
        "category": "Historical Incident Records",
        "equipment_type": "Plant Operations & Motors",
        "content": (
            "A previous motor overheating incident was caused by worn bearings and "
            "insufficient lubrication. The issue was resolved by replacing the bearings "
            "and applying proper lubrication. Root cause analysis indicated thermal degradation "
            "of grease due to extended operating intervals beyond scheduled maintenance."
        ),
        "typical_causes": [
            "Extended operation past scheduled maintenance window",
            "Thermal degradation of grease",
            "Secondary bearing cage breakdown"
        ],
        "recommended_actions": [
            "Review past maintenance logs for overdue service intervals",
            "Replace worn bearing assembly and flush grease cavity",
            "Enforce strict operating-hour tracking and thermal logging"
        ],
        "keywords": ["incident", "overheating", "bearing", "lubrication", "root cause", "history"]
    }
]


def get_all_documents() -> List[Dict[str, Any]]:
    """Returns the full list of industrial knowledge base documents."""
    return INDUSTRIAL_DOCUMENTS


def get_document_chunks() -> List[Dict[str, Any]]:
    """
    Returns chunked documents suitable for embedding and vector indexing.
    Each chunk retains document metadata for source transparency.
    """
    chunks = []
    for doc in INDUSTRIAL_DOCUMENTS:
        # Full content representation for vector indexing
        searchable_text = f"{doc['title']} ({doc['equipment_type']}): {doc['content']}"
        chunks.append({
            "id": doc["id"],
            "title": doc["title"],
            "category": doc["category"],
            "equipment_type": doc["equipment_type"],
            "chunk_text": searchable_text,
            "raw_content": doc["content"],
            "typical_causes": doc.get("typical_causes", []),
            "recommended_actions": doc.get("recommended_actions", []),
        })
    return chunks
