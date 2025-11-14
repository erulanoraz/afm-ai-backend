# app/search/vector_client.py
import weaviate

class VectorClient:
    def __init__(self):
        self.client = weaviate.Client("http://localhost:8080")

    def search(self, vector, top_k=100):
        res = (
            self.client.query
            .get("Chunk", ["chunk_id", "text", "file_id"])
            .with_near_vector({"vector": vector})
            .with_limit(top_k)
            .do()
        )
        hits = []
        for d in res["data"]["Get"]["Chunk"]:
            hits.append({
                "id": d["chunk_id"],
                "text": d["text"],
                "score": 1.0  # Weaviate возвращает cosine similarity
            })
        return hits
