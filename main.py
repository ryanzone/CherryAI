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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ✅ FINAL WORKING MODEL + ENDPOINT
GEMINI_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"

# ── Models ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    assets: List[Any] = []

# ── Asset Fetcher ─────────────────────────────────────────────

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

async def fetch_asset(client: httpx.AsyncClient, url: str) -> dict | None:
    if not isinstance(url, str) or not url.startswith("http"):
        return {"text": f"[asset: {url}]"}
    try:
        r = await client.get(url, timeout=10)
        ct = r.headers.get("content-type", "").split(";")[0].lower()

        if ct in IMAGE_TYPES:
            return {
                "inlineData": {
                    "mimeType": ct,
                    "data": base64.b64encode(r.content).decode()
                }
            }

        return {"text": r.text[:8000]}

    except Exception as e:
        return {"text": f"[error loading asset: {e}]"}

# ── Gemini Call ───────────────────────────────────────────────

async def call_gemini(parts: list) -> str:
    payload = {
        "contents": [{
            "parts": parts
        }]
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload
        )

        data = r.json()

    if "candidates" not in data:
        raise HTTPException(status_code=500, detail=str(data))

    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

# ── Routes ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ok"}

@app.post("/v1/answer")
async def answer(request: QueryRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query empty")

    parts = []

    # assets
    if request.assets:
        async with httpx.AsyncClient() as client:
            res = await asyncio.gather(*[fetch_asset(client, a) for a in request.assets])
        parts.extend([p for p in res if p])

    # 🔥 IMPORTANT PROMPT CONTROL
    parts.append({
        "text": "Answer directly. No explanation. If numeric return only number: " + query
    })

    output = await call_gemini(parts)

    return {"output": output}

# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)