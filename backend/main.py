from fastapi import FastAPI, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import json
from matcher import match_candidates
import uvicorn

app = FastAPI(title="MatchCV")


with open("../data/vacancies.json", "r", encoding="utf-8") as f:
    vacancies = json.load(f)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_file.read_text(encoding="utf-8"))

@app.post("/search")
async def search(vacancy_id: str = Form(...), top_k: int = Form(10)):
    vac = next((v for v in vacancies if v["id"] == vacancy_id), None)
    if not vac:
        return {"error": "Vacancy not found"}
    results = await match_candidates(vac, top_k)
    return {"vacancy": vac, "candidates": results}

@app.get("/vacancies")
async def list_vacancies():
    return vacancies


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)