import os
import re
import base64
import httpx
import json
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any

app = FastAPI()

# ── Config ─────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.2-90b-vision-preview"

# ── Models ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    assets: List[Any] = []

# ── Asset Fetcher ─────────────────────────────────────────────

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

async def fetch_asset(client: httpx.AsyncClient, url: str) -> dict | None:
    if not isinstance(url, str) or not url.startswith("http"):
        return {"type": "text", "text": f"[asset: {url}]"}
    try:
        r = await client.get(url, timeout=10)
        ct = r.headers.get("content-type", "").split(";")[0].lower()

        if ct in IMAGE_TYPES:
            b64 = base64.b64encode(r.content).decode()
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{ct};base64,{b64}"
                }
            }

        return {"type": "text", "text": r.text[:8000]}

    except Exception as e:
        return {"type": "text", "text": f"[error loading asset: {e}]"}

# ── Groq Call ─────────────────────────────────────────────────

async def call_groq(content: list) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.2,
        "max_tokens": 256
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            GROQ_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
        )
        
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
            
        data = r.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"Groq parse error: {e}. Data: {json.dumps(data)}")

# ── Routes ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ok"}

@app.post("/v1/answer")
async def answer(request: QueryRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query empty")

    content_parts = []

    # assets
    if request.assets:
        async with httpx.AsyncClient() as client:
            res = await asyncio.gather(*[fetch_asset(client, a) for a in request.assets])
        content_parts.extend([p for p in res if p])

    # 🔥 IMPORTANT PROMPT CONTROL
    prompt = "Answer directly. No explanation. If numeric return only number: " + query
    content_parts.append({"type": "text", "text": prompt})

    output = await call_groq(content_parts)

    return {"output": output}

# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)