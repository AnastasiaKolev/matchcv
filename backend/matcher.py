import numpy as np
from typing import List, Dict
from embedding_service import EmbeddingService
from opensearchpy import OpenSearch
import asyncio
from vllm_client import generate_comment

OPENSEARCH_HOST = "opensearch"
INDEX_NAME = "resumes"

embed_service = EmbeddingService()


def normalize_score(scores):
    min_s, max_s = min(scores), max(scores)
    return [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]


async def match_candidates(vacancy: dict, top_k: int = 10) -> List[Dict]:
    client = OpenSearch(f"http://{OPENSEARCH_HOST}:9200")
    vac_embedding = embed_service.encode([vacancy["text"]])[0].tolist()

    query = {
        "size": 100,
        "query": {
            "script_score": {
                "query": {
                    "match": {"text": vacancy["text"]}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0 + _score * 0.3",
                    "params": {"query_vector": vac_embedding}
                }
            }
        }
    }
    response = client.search(index=INDEX_NAME, body=query)
    candidates = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        vec_sim = hit["_score"]  # уже суммарный
        skills_match = len(set(src["skills"]) & set(vacancy["required_skills"])) / max(len(vacancy["required_skills"]),
                                                                                       1)
        opt_match = len(set(src["skills"]) & set(vacancy["optional_skills"])) / max(len(vacancy["optional_skills"]), 1)
        exp_diff = abs(src["experience_years"] - vacancy["min_experience"])
        exp_score = 1.0 if src["experience_years"] >= vacancy["min_experience"] else max(0, 1 - exp_diff / 5)

        if src["experience_years"] > vacancy["min_experience"] + 3:
            exp_score *= 0.9

        final = vec_sim * 0.4 + skills_match * 0.4 + opt_match * 0.1 + exp_score * 0.1
        candidates.append({
            "id": src["id"],
            "resume_id": src["id"],
            "text": src["text"],
            "skills": src["skills"],
            "experience_years": src["experience_years"],
            "score": final,
            "vec_sim": vec_sim,
            "skills_match": skills_match,
            "exp_score": exp_score
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:top_k]

    comments = await asyncio.gather(*[generate_comment(vacancy["text"], c["text"]) for c in top])
    for c, comm in zip(top, comments):
        c["comment"] = comm

    return top