# app/search/fulltext_client.py
from opensearchpy import OpenSearch
from app.utils.config import settings

class FulltextClient:
    def __init__(self):
        self.client = OpenSearch(
            settings.ES_URL,  # добавим это в .env
            http_auth=(settings.ES_USER, settings.ES_PASS),
            verify_certs=False
        )

    def search(self, query, top_k=100):
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "metadata"]
                }
            }
        }
        res = self.client.search(index="chunks", body=body)
        hits = [
            {"id": h["_id"], "text": h["_source"]["text"], "score": h["_score"]}
            for h in res["hits"]["hits"]
        ]
        return hits
