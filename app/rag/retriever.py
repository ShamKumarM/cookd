import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


class CookdRetriever:

    def __init__(self):

        root = Path(__file__).resolve().parents[2]

        chroma_path = os.getenv(
            "CHROMA_PATH",
            str(root / "chroma_data")
        )

        collection_name = os.getenv(
            "CHROMA_COLLECTION",
            "cookd_knowledge"
        )

        embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=chroma_path
        )

        self.collection = (
            self.client.get_or_create_collection(
                collection_name
            )
        )

        self.embedding_model_name = (
            embedding_model_name
        )

        # Do NOT load SentenceTransformer here
        self.model = None

    def _get_model(self):

        if self.model is None:

            print(
                "Loading embedding model:",
                self.embedding_model_name
            )

            self.model = SentenceTransformer(
                self.embedding_model_name,
                device="cpu"
            )

            print("Embedding model loaded.")

        return self.model

    def search(
        self,
        query,
        top_k=5,
        doc_type=None
    ):

        model = self._get_model()

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": top_k
        }

        if doc_type:
            kwargs["where"] = {
                "type": doc_type
            }

        result = self.collection.query(
            **kwargs
        )

        documents = result.get(
            "documents",
            [[]]
        )[0]

        metadatas = result.get(
            "metadatas",
            [[]]
        )[0]

        distances = result.get(
            "distances",
            [[]]
        )[0]

        return [
            {
                "text": document,
                "metadata": (
                    metadatas[i]
                    if i < len(metadatas)
                    else {}
                ),
                "distance": (
                    distances[i]
                    if i < len(distances)
                    else None
                )
            }
            for i, document in enumerate(documents)
        ]