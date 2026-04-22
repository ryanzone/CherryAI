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

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    assets: List[Any] = []

class QueryResponse(BaseModel):
    output: str

# ── Asset Fetcher ─────────────────────────────────────────────────────────────

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

async def fetch_asset(client: httpx.AsyncClient, url: str) -> dict | None:
    if not isinstance(url, str) or not url.startswith("http"):
        return {"text": f"[asset: {url}]"}
    try:
        r = await client.get(url, timeout=10, follow_redirects=True)
        r.raise_for_status()
        ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
        ext = re.search(r'\.\w+$', url.split("?")[0])
        ext = ext.group(0).lower() if ext else ""

        if ct in IMAGE_TYPES or ext in IMAGE_EXTS:
            b64 = base64.b64encode(r.content).decode()
            mime = ct if ct in IMAGE_TYPES else "image/jpeg"
            return {"inline_data": {"mime_type": mime, "data": b64}}

        if ct == "application/pdf" or ext == ".pdf":
            b64 = base64.b64encode(r.content).decode()
            return {"inline_data": {"mime_type": "application/pdf", "data": b64}}

        text = r.text[:12000]
        return {"text": f"[Asset from {url}]:\n{text}"}

    except Exception as e:
        return {"text": f"[Could not fetch {url}: {e}]"}

# ── Gemini Call ───────────────────────────────────────────────────────────────

async def call_gemini(parts: list) -> str:
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
        "systemInstruction": {
            "parts": [{
                "text": (
                    "You are a precise question-answering assistant. "
                    "Answer the user's query directly and concisely. "
                    "No markdown, no bullet points unless explicitly asked for a list. "
                    "No preamble, no trailing explanation. "
                    "If the query is about provided assets/documents/images, analyse and use them. "
                    "Single sentence answers when possible. "
                    "No newlines in your response."
                )
            }]
        }
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        r.raise_for_status()
        data = r.json()

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip().replace("\n", " ")
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail=f"Gemini error: {json.dumps(data)}")

# ── Main Endpoint ─────────────────────────────────────────────────────────────

@app.post("/v1/answer", response_model=QueryResponse)
async def answer(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    parts = []

    if request.assets:
        async with httpx.AsyncClient() as client:
            asset_parts = await asyncio.gather(
                *[fetch_asset(client, url) for url in request.assets]
            )
        for p in asset_parts:
            if p:
                parts.append(p)

    parts.append({"text": query})

    output = await call_gemini(parts)
    return QueryResponse(output=output)

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "ok"}

@app.post("/v1/answer")
async def answer(request: QueryRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    parts = []

    if request.assets:
        async with httpx.AsyncClient() as client:
            asset_parts = await asyncio.gather(
                *[fetch_asset(client, url) for url in request.assets]
            )
        for p in asset_parts:
            if p:
                parts.append(p)

    parts.append({"text": query})

    output = await call_gemini(parts)

    return {"output": output}

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
