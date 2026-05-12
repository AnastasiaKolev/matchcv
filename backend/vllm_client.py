import aiohttp
import json

VLLM_URL = "http://vllm:8000/generate"

async def generate_comment(vacancy: str, resume: str) -> dict:
    prompt = f"""<|im_start|>system
Ты — ассистент рекрутера. Проанализируй резюме кандидата относительно вакансии и напиши краткий комментарий на русском в формате JSON с ключами:
strengths (строка), skills_match_details (строка), risks (строка).
<|im_end|>
<|im_start|>user
Вакансия: {vacancy}
Резюме: {resume}
<|im_end|>
<|im_start|>assistant
"""
    payload = {
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.2,
        "stop": ["<|im_end|>"]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(VLLM_URL, json=payload) as resp:
            result = await resp.json()
            text = result["text"][0] if isinstance(result["text"], list) else result["text"]
            try:
                return json.loads(text)
            except:
                return {"strengths": "", "skills_match_details": "", "risks": text}