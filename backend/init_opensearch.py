from opensearchpy import OpenSearch, helpers
import json
import os
from embedding_service import EmbeddingService

INDEX = "resumes"
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
client = OpenSearch(f"http://{OPENSEARCH_HOST}:9200")

mapping = {
    "settings": {"number_of_shards": 1},
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "russian"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil"
                }
            }
        }
    }
}

if client.indices.exists(index=INDEX):
    client.indices.delete(index=INDEX)
client.indices.create(index=INDEX, body=mapping)

embedder = EmbeddingService()
with open("data/resumes.json", "r", encoding="utf-8") as f:
    resumes = json.load(f)

actions = []
for resume in resumes:
    emb = embedder.encode([resume["text"]])[0].tolist()
    actions.append({
        "_index": INDEX,
        "_id": resume["id"],
        "_source": {
            "id": resume["id"],
            "text": resume["text"],
            "skills": resume["skills"],
            "experience_years": resume["experience_years"],
            "education": resume["education"],
            "embedding": emb
        }
    })

success, failed = helpers.bulk(client, actions, stats_only=True)
print(f"Успешно загружено: {success}, ошибок: {failed}")