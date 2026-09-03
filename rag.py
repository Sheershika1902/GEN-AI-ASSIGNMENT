"""
rag.py - Retrieval-Augmented Generation (RAG) Pipeline
Uses SentenceTransformers (all-MiniLM-L6-v2) and FAISS for vector similarity search.
Generates grounded diagnostic guidance from retrieved technical documents.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from documents import get_document_chunks

# Try importing dependencies with helpful error messages
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None


class IndustrialRAGPipeline:
    """
    RAG Pipeline for Industrial Diagnostics.
    Architecture:
    Documents -> Chunking -> Embeddings (all-MiniLM-L6-v2) -> FAISS IndexFlatIP -> Similarity Search -> Grounded Synthesis
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.35):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.chunks = get_document_chunks()
        self.model = None
        self.index = None
        self.embeddings = None
        self.is_initialized = False
        self.initialization_error = None
        
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Loads embedding model and builds the FAISS vector index."""
        try:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is not installed. Run: pip install sentence-transformers")
            if faiss is None:
                raise ImportError("faiss-cpu is not installed. Run: pip install faiss-cpu")

            # 1. Load Sentence Transformer model
            self.model = SentenceTransformer(self.model_name)

            # 2. Extract texts to embed
            texts = [chunk["chunk_text"] for chunk in self.chunks]

            # 3. Compute Embeddings
            raw_embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            
            # Normalize embeddings to unit length for Cosine Similarity via Inner Product
            self.embeddings = raw_embeddings.astype("float32")
            faiss.normalize_L2(self.embeddings)

            # 4. Build FAISS Index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product on normalized vectors = Cosine Similarity
            self.index.add(self.embeddings)

            self.is_initialized = True
        except Exception as e:
            self.is_initialized = False
            self.initialization_error = str(e)

    def search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves Top-K relevant documents from FAISS index for a given query.
        Returns:
            Dict containing retrieved chunks, similarity scores, and low_evidence flag.
        """
        if not self.is_initialized:
            # Fallback for when dependencies or model are loading
            return self._keyword_fallback_search(query, top_k)

        # 1. Encode query and normalize for cosine similarity
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)

        # 2. Perform FAISS search
        k = min(top_k, len(self.chunks))
        similarities, indices = self.index.search(query_embedding, k)

        retrieved_results = []
        for score, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            # Normalize score to 0.0 - 1.0 range
            sim_score = max(0.0, min(1.0, float(score)))
            chunk["similarity_score"] = sim_score
            chunk["confidence_percent"] = round(sim_score * 100, 1)
            retrieved_results.append(chunk)

        # 3. Evaluate Low Evidence condition
        top_score = retrieved_results[0]["similarity_score"] if retrieved_results else 0.0
        is_low_evidence = top_score < self.similarity_threshold

        return {
            "query": query,
            "top_k": top_k,
            "results": retrieved_results,
            "top_score": top_score,
            "is_low_evidence": is_low_evidence,
            "threshold_used": self.similarity_threshold
        }

    def _keyword_fallback_search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Graceful fallback if vector search fails or dependencies are missing."""
        query_words = set(query.lower().split())
        scored_chunks = []

        for chunk in self.chunks:
            text = (chunk["chunk_text"] + " " + chunk["category"]).lower()
            matches = sum(1 for word in query_words if word in text)
            score = min(1.0, matches / max(1, len(query_words)))
            item = chunk.copy()
            item["similarity_score"] = float(score)
            item["confidence_percent"] = round(score * 100, 1)
            scored_chunks.append(item)

        scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = scored_chunks[:top_k]
        top_score = results[0]["similarity_score"] if results else 0.0

        return {
            "query": query,
            "top_k": top_k,
            "results": results,
            "top_score": top_score,
            "is_low_evidence": top_score < 0.2,
            "threshold_used": self.similarity_threshold,
            "fallback_mode": True
        }

    def generate_diagnostic_guidance(
        self,
        query: str,
        multimodal_context: str,
        retrieved_data: Dict[str, Any],
        image_observations: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generates grounded, transparent diagnostic guidance based on retrieved documents
        and multimodal context without hallucination.
        """
        if retrieved_data.get("is_low_evidence", False):
            return {
                "status": "low_evidence",
                "summary": "Insufficient relevant evidence was found in the industrial knowledge base for this specific query.",
                "possible_causes": [
                    "Query does not match registered equipment manuals or standard failure modes.",
                    "Equipment type or anomaly terminology may not be covered in the current knowledge base."
                ],
                "recommended_actions": [
                    "Provide more detailed technical specifics (equipment name, operational symptoms, error codes).",
                    "Consult on-site mechanical/electrical maintenance engineers.",
                    "Verify manufacturer technical service bulletin or physical maintenance manual."
                ],
                "evidence_sources": [],
                "responsible_ai_note": "No conclusive diagnostic claims are made due to low retrieval confidence."
            }

        retrieved_docs = retrieved_data.get("results", [])
        
        # Aggregate possible causes from retrieved documents
        possible_causes = []
        recommended_actions = []
        evidence_sources = []

        for doc in retrieved_docs:
            evidence_sources.append({
                "id": doc["id"],
                "title": doc["title"],
                "equipment_type": doc["equipment_type"],
                "category": doc["category"],
                "confidence_percent": doc["confidence_percent"],
                "excerpt": doc["raw_content"]
            })
            
            for cause in doc.get("typical_causes", []):
                if cause not in possible_causes:
                    possible_causes.append(cause)
                    
            for action in doc.get("recommended_actions", []):
                if action not in recommended_actions:
                    recommended_actions.append(action)

        # Modality correlation: Incorporate image observations if present
        if image_observations and len(image_observations) > 0 and "No Visible Damage" not in image_observations:
            for obs in image_observations:
                obs_cause = f"Visual inspection anomaly: Identified evidence of '{obs}' on component exterior"
                if obs_cause not in possible_causes:
                    possible_causes.insert(0, obs_cause)
                    
                obs_action = f"Perform localized physical inspection and non-destructive testing for verified '{obs}'"
                if obs_action not in recommended_actions:
                    recommended_actions.insert(0, obs_action)

        # Build grounded response summary
        primary_doc = retrieved_docs[0]["title"] if retrieved_docs else "Knowledge Base"
        summary = (
            f"Based on available technical documentation (primarily {primary_doc}) and the provided multimodal symptoms, "
            f"the system identified {len(possible_causes)} potential root causes and {len(recommended_actions)} actionable inspection steps."
        )

        return {
            "status": "success",
            "summary": summary,
            "possible_causes": possible_causes[:5],  # Top 5 most relevant causes
            "recommended_actions": recommended_actions[:5],  # Top 5 actionable steps
            "evidence_sources": evidence_sources,
            "responsible_ai_note": (
                "Guidance is grounded strictly in retrieved standard operating procedures and user-provided observations. "
                "These findings represent potential diagnostic hypotheses and require physical validation by certified personnel."
            )
        }
