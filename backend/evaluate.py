import json
import asyncio
import numpy as np
from matcher import match_candidates
import os

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    dcg = dcg_at_k(y_true_sorted, k)
    ideal = dcg_at_k(sorted(y_true, reverse=True), k)
    return dcg / ideal if ideal else 0.0

async def evaluate():
    with open("../data/vacancies.json", "r") as f:
        vacancies = json.load(f)
    with open("../data/resumes.json", "r") as f:
        resumes = json.load(f)

    ndcg_list = []
    prec_list = []
    rec_list = []
    mrr_list = []

    for vac in vacancies:
        relevant_ids = set()
        for r in resumes:
            if set(vac["required_skills"]).issubset(set(r["skills"])) and r["experience_years"] >= vac["min_experience"]:
                relevant_ids.add(r["id"])

        candidates = await match_candidates(vac, top_k=20)
        cand_ids = [c["id"] for c in candidates]
        y_true = np.array([1 if cid in relevant_ids else 0 for cid in cand_ids])
        y_score = np.array([c["score"] for c in candidates])

        ndcg = ndcg_at_k(y_true, y_score, 10)
        ndcg_list.append(ndcg)

        top10 = cand_ids[:10]
        tp = len(set(top10) & relevant_ids)
        prec = tp / 10
        rec = tp / len(relevant_ids) if relevant_ids else 0
        prec_list.append(prec)
        rec_list.append(rec)

        for rank, cid in enumerate(cand_ids, 1):
            if cid in relevant_ids:
                mrr_list.append(1.0 / rank)
                break
        else:
            mrr_list.append(0.0)

    print(f"NDCG@10: {np.mean(ndcg_list):.3f}")
    print(f"Precision@10: {np.mean(prec_list):.3f}")
    print(f"Recall@10: {np.mean(rec_list):.3f}")
    print(f"MRR: {np.mean(mrr_list):.3f}")

if __name__ == "__main__":
    asyncio.run(evaluate())