import os, json, numpy as np
from typing import List, Dict
from embedding_service import EmbeddingService

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
embed_service = EmbeddingService()

class InMemoryMatcher:
    def __init__(self):
        with open(os.path.join(DATA_DIR, "resumes.json"), "r", encoding="utf-8") as f:
            self.resumes = json.load(f)
        self.texts = [r["text"] for r in self.resumes]
        self.embeddings = embed_service.encode(self.texts)
        self.skills_list = [r["skills"] for r in self.resumes]
        self.exp_list = [r["experience_years"] for r in self.resumes]

    def search(self, vac_embedding, vac, top_k=10):
        # косинусное сходство (векторы уже нормализованы)
        sims = np.dot(self.embeddings, vac_embedding)
        order = np.argsort(sims)[::-1][:top_k*2]  # берём побольше для реранкинга
        candidates = []
        for idx in order:
            r = self.resumes[idx]
            skills_match = len(set(r["skills"]) & set(vac["required_skills"])) / max(len(vac["required_skills"]), 1)
            opt_match = len(set(r["skills"]) & set(vac["optional_skills"])) / max(len(vac["optional_skills"]), 1)
            exp_diff = abs(r["experience_years"] - vac["min_experience"])
            exp_score = 1.0 if r["experience_years"] >= vac["min_experience"] else max(0, 1 - exp_diff/5)
            if r["experience_years"] > vac["min_experience"] + 3:
                exp_score *= 0.9
            final = sims[idx]*0.4 + skills_match*0.4 + opt_match*0.1 + exp_score*0.1
            candidates.append({
                "id": r["id"],
                "resume_id": r["id"],
                "text": r["text"],
                "skills": r["skills"],
                "experience_years": r["experience_years"],
                "score": final,
                "vec_sim": float(sims[idx]),
                "skills_match": skills_match,
                "exp_score": exp_score
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:top_k]
        for c in top:
            strengths = []; risks = []
            if c["skills_match"] >= 0.8: strengths.append("Полный набор обязательных навыков")
            if c["exp_score"] >= 0.9: strengths.append("Достаточный опыт")
            if c["skills_match"] < 0.5: risks.append("Не хватает ключевых навыков")
            if c["exp_score"] < 0.8: risks.append("Опыт ниже требуемого")
            c["comment"] = {
                "strengths": "; ".join(strengths) or "Нет явных преимуществ",
                "risks": "; ".join(risks) or "Риски отсутствуют"
            }
        return top

matcher = InMemoryMatcher()

async def match_candidates(vacancy: dict, top_k: int = 10) -> List[Dict]:
    vac_emb = embed_service.encode([vacancy["text"]])[0]
    return matcher.search(vac_emb, vacancy, top_k)