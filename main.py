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

# ── Config ─────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ✅ WORKING MODEL
MODEL = "llama-3.1-8b-instant"

# ── Models ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    assets: List[Any] = []

# ── Asset Fetcher ──────────────────────────────────────

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

async def fetch_asset(client: httpx.AsyncClient, url: str) -> dict | None:
    if not isinstance(url, str) or not url.startswith("http"):
        return {"type": "text", "text": f"[asset: {url}]"}
    try:
        r = await client.get(url, timeout=10)
        ct = r.headers.get("content-type", "").split(";")[0].lower()
        ext = os.path.splitext(url)[1].lower()

        if ct in IMAGE_TYPES or ext in IMAGE_EXTS:
            b64 = base64.b64encode(r.content).decode()
            mime = ct if ct in IMAGE_TYPES else "image/jpeg"
            return {
                "type": "text",
                "text": "[Image provided]"
            }

        text = r.text[:8000]
        return {"type": "text", "text": text}

    except Exception as e:
        return {"type": "text", "text": f"[error loading asset: {e}]"}

# ── Groq Call ──────────────────────────────────────────

async def call_groq(parts: list) -> str:
    # Convert parts → plain text
    text_input = ""
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            text_input += p.get("text", "") + "\n"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Answer directly. If numeric return only number. No explanation."
            },
            {
                "role": "user",
                "content": text_input
            }
        ],
        "temperature": 0.2,
        "max_tokens": 256
    }

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            GROQ_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
        )

        data = r.json()

    if "choices" not in data:
        raise HTTPException(status_code=500, detail=str(data))

    return data["choices"][0]["message"]["content"].strip()

# ── Routes ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ok"}

@app.post("/v1/answer")
async def answer(request: QueryRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query empty")

    parts = []

    # Fetch assets
    if request.assets:
        async with httpx.AsyncClient() as client:
            res = await asyncio.gather(*[fetch_asset(client, a) for a in request.assets])
        parts.extend([p for p in res if p])

    # Add query with strict instruction
    parts.append({
        "type": "text",
        "text": "Answer directly. No explanation. If numeric return only number: " + query
    })

    output = await call_groq(parts)

    return {"output": output}

# ── Run ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)