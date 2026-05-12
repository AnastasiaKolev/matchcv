import asyncio
import json
import sys
import os
import numpy as np
from sklearn.metrics import ndcg_score
from matcher import match_candidates
from opensearchpy import OpenSearch

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
INDEX = "resumes"

async def evaluate():
    with open("../data/vacancies.json", "r") as f:
        vacancies = json.load(f)
    with open("../data/resumes.json", "r") as f:
        resumes = json.load(f)

    client = OpenSearch(f"http://{OPENSEARCH_HOST}:9200")

    ndcg_scores = []
    precision_scores = []
    recall_scores = []
    mrr_scores = []

    for vac in vacancies:
        relevant_ids = set()
        for r in resumes:
            if set(vac["required_skills"]).issubset(set(r["skills"])) and r["experience_years"] >= vac["min_experience"]:
                relevant_ids.add(r["id"])

        candidates = await match_candidates(vac, top_k=20)
        candidate_ids = [c["id"] for c in candidates]
        y_true = [3 if cid in relevant_ids else 0 for cid in candidate_ids]
        y_score = [c["score"] for c in candidates]

        if len(y_true) >= 10:
            ndcg = ndcg_score([y_true[:10]], [y_score[:10]])
        else:
            ndcg = ndcg_score([y_true], [y_score])
        ndcg_scores.append(ndcg)

        top10 = candidate_ids[:10]
        tp = len(set(top10) & relevant_ids)
        precision = tp / 10
        recall = tp / len(relevant_ids) if relevant_ids else 0
        precision_scores.append(precision)
        recall_scores.append(recall)

        for rank, cid in enumerate(candidate_ids, 1):
            if cid in relevant_ids:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

    print(f"NDCG@10: {np.mean(ndcg_scores):.3f}")
    print(f"Precision@10: {np.mean(precision_scores):.3f}")
    print(f"Recall@10: {np.mean(recall_scores):.3f}")
    print(f"MRR: {np.mean(mrr_scores):.3f}")

if __name__ == "__main__":
    asyncio.run(evaluate())