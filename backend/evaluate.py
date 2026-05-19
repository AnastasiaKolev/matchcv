import json
import asyncio
import numpy as np
from pathlib import Path

from backend.matcher import match_candidates


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def dcg_at_k(relevances, k):
    """
    DCG@k for graded relevance.
    Чем выше релевантный кандидат в списке, тем больше вклад.
    """
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0

    gains = (2 ** rel - 1)
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances, k):
    """
    NDCG@k = DCG текущего ранжирования / идеальный DCG.
    Использует graded relevance: 0, 1, 2, 3.
    """
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def relevance_score(vacancy, resume):
    """
    Более мягкая оценка релевантности кандидата вакансии.

    Возвращает:
    0 — не подходит
    1 — слабое соответствие
    2 — частичное/среднее соответствие
    3 — сильное соответствие
    """

    required = set(vacancy.get("required_skills", []))
    optional = set(vacancy.get("optional_skills", []))
    skills = set(resume.get("skills", []))

    # 1. Совпадение обязательных навыков
    if required:
        required_match = len(required & skills) / len(required)
    else:
        required_match = 1.0

    # 2. Совпадение желательных навыков
    if optional:
        optional_match = len(optional & skills) / len(optional)
    else:
        optional_match = 0.0

    # 3. Соответствие опыту
    exp = resume.get("experience_years", 0)
    min_exp = vacancy.get("min_experience", 0)

    if exp >= min_exp:
        exp_score = 1.0
    elif exp >= max(0, min_exp - 1):
        exp_score = 0.8
    elif exp >= max(0, min_exp - 2):
        exp_score = 0.5
    else:
        exp_score = 0.0

    # 4. Общий soft score релевантности
    soft_score = (
        0.65 * required_match +
        0.20 * optional_match +
        0.15 * exp_score
    )

    # 5. Переводим soft score в graded relevance
    if required_match >= 0.8 and exp_score >= 0.8:
        return 3
    elif soft_score >= 0.60 and required_match >= 0.50:
        return 2
    elif soft_score >= 0.35:
        return 1
    else:
        return 0


async def evaluate():
    with open(DATA_DIR / "vacancies.json", "r", encoding="utf-8") as f:
        vacancies = json.load(f)

    with open(DATA_DIR / "resumes.json", "r", encoding="utf-8") as f:
        resumes = json.load(f)

    resumes_by_id = {r["id"]: r for r in resumes}

    ndcg_list = []
    precision_list = []
    recall_list = []
    mrr_list = []

    k = 10
    relevance_threshold = 2

    for vacancy in vacancies:
        # Считаем graded relevance для всех резюме
        all_relevances = {
            r["id"]: relevance_score(vacancy, r)
            for r in resumes
        }

        # Релевантными для Precision/Recall/MRR считаем кандидатов с relevance >= 2
        relevant_ids = {
            resume_id
            for resume_id, rel in all_relevances.items()
            if rel >= relevance_threshold
        }

        candidates = await match_candidates(vacancy, top_k=20)
        cand_ids = [c["id"] for c in candidates]

        ranked_relevances = [
            all_relevances.get(cid, 0)
            for cid in cand_ids
        ]

        # NDCG@10 по graded relevance
        ndcg_list.append(ndcg_at_k(ranked_relevances, k))

        # Precision@10
        top_k_ids = cand_ids[:k]
        tp = len(set(top_k_ids) & relevant_ids)
        precision_list.append(tp / k)

        # Recall@10
        if relevant_ids:
            recall_list.append(tp / len(relevant_ids))
        else:
            recall_list.append(0.0)

        # MRR
        reciprocal_rank = 0.0
        for rank, cid in enumerate(cand_ids, start=1):
            if cid in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break

        mrr_list.append(reciprocal_rank)

    print(f"NDCG@10: {np.mean(ndcg_list):.3f}")
    print(f"Precision@10: {np.mean(precision_list):.3f}")
    print(f"Recall@10: {np.mean(recall_list):.3f}")
    print(f"MRR: {np.mean(mrr_list):.3f}")


if __name__ == "__main__":
    asyncio.run(evaluate())