import faiss, pickle, time, numpy as np
from sentence_transformers import SentenceTransformer
import torch

class RAGAgent:
    """
    RAG Agent - FAISS vector search over MIMIC-IV discharge notes.
    Target: < 100ms per query.
    """
    def __init__(self, faiss_path, meta_path,
                 embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 n_probe=10, top_k=5):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index = faiss.read_index(faiss_path)
        self.index.nprobe = n_probe
        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(embed_model_name, device=device)
        self.top_k = top_k

    def retrieve(self, query_text):
        t0  = time.perf_counter()
        vec = self.model.encode([query_text[:500]], convert_to_numpy=True).astype("float32")
        dists, idxs = self.index.search(vec, self.top_k)
        results = []
        for d, i in zip(dists[0], idxs[0]):
            if i < 0:
                continue
            c = self.chunks[i]
            results.append({
                "text": c["text"], "note_id": c["note_id"],
                "subject_id": c["subject_id"],
                "similarity": round(1.0 / (1.0 + float(d)), 4),
            })
        context = "\n\n---\n\n".join(
            f"[Case {i+1} | sim={r['similarity']:.3f}]\n{r['text']}"
            for i, r in enumerate(results)
        )
        return {"retrieved": results, "context": context, "n": len(results),
                "latency_ms": round((time.perf_counter()-t0)*1000, 2)}
