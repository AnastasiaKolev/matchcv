import json
import random

SKILLS_POOL = [
    "Python", "SQL", "Docker", "Kubernetes", "Machine Learning",
    "Deep Learning", "PyTorch", "TensorFlow", "Agile", "Scrum",
    "Linux", "Git", "CI/CD", "AWS", "GCP", "Data Engineering",
    "Spark", "Hadoop", "Airflow", "Tableau"
]

def generate_resume(idx: int, position: str) -> dict:
    skills = random.sample(SKILLS_POOL, random.randint(3, 7))
    exp_years = random.randint(1, 10)
    education = random.choice(["Бакалавр", "Магистр", "Высшее"])
    text = f"Резюме кандидата #{idx}\nДолжность: {position}\nОпыт: {exp_years} лет\n" \
           f"Навыки: {', '.join(skills)}\nОбразование: {education}\n" \
           f"Дополнительная информация: {random.choice(['Командный игрок', 'Умеет работать в стрессовых ситуациях', 'Любит учиться'])}"
    return {
        "id": str(idx),
        "position": position,
        "text": text,
        "skills": skills,
        "experience_years": exp_years,
        "education": education
    }

def generate_vacancy(idx: int, title: str) -> dict:
    req_skills = random.sample(SKILLS_POOL, random.randint(4, 6))
    opt_skills = random.sample([s for s in SKILLS_POOL if s not in req_skills], random.randint(1, 3))
    min_exp = random.randint(1, 5)
    text = f"Вакансия: {title}\nТребуемый опыт: от {min_exp} лет\n" \
           f"Обязательные навыки: {', '.join(req_skills)}\n" \
           f"Желательные навыки: {', '.join(opt_skills)}\n" \
           f"Обязанности: Разработка, тестирование, поддержка.\nУсловия: офис, соцпакет."
    return {
        "id": str(idx),
        "title": title,
        "text": text,
        "required_skills": req_skills,
        "optional_skills": opt_skills,
        "min_experience": min_exp
    }

def create_dataset(num_resumes=500, num_vacancies=20):
    positions = ["Data Scientist", "Python Backend Developer", "DevOps", "Product Manager"]
    resumes = [generate_resume(i, random.choice(positions)) for i in range(num_resumes)]
    vacancies = [generate_vacancy(i, random.choice(positions)) for i in range(num_vacancies)]
    with open("data/resumes.json", "w", encoding="utf-8") as f:
        json.dump(resumes, f, ensure_ascii=False, indent=2)
    with open("data/vacancies.json", "w", encoding="utf-8") as f:
        json.dump(vacancies, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    create_dataset()