import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

class CookdRetriever:
    def __init__(self):
        root=Path(__file__).resolve().parents[2]
        self.client=chromadb.PersistentClient(path=os.getenv("CHROMA_PATH",str(root/"chroma_data")))
        self.collection=self.client.get_or_create_collection(os.getenv("CHROMA_COLLECTION","cookd_knowledge"))
        self.model=SentenceTransformer(os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2"))

    def search(self,query,top_k=5,doc_type=None):
        q=self.model.encode([query],normalize_embeddings=True).tolist()
        kw={"query_embeddings":q,"n_results":top_k}
        if doc_type: kw["where"]={"type":doc_type}
        r=self.collection.query(**kw)
        return [{"text":d,"metadata":r["metadatas"][0][i],
                 "distance":r["distances"][0][i] if r.get("distances") else None}
                for i,d in enumerate(r["documents"][0])]
