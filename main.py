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
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. API calls will fail.")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.0-pro:generateContent"
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
            return {"inlineData": {"mimeType": mime, "data": b64}}

        if ct == "application/pdf" or ext == ".pdf":
            b64 = base64.b64encode(r.content).decode()
            return {"inlineData": {"mimeType": "application/pdf", "data": b64}}

        text = r.text[:12000]
        return {"text": f"[Asset from {url}]:\n{text}"}

    except Exception as e:
        return {"text": f"[Could not fetch {url}: {e}]"}

# ── Gemini Call ───────────────────────────────────────────────────────────────

async def call_gemini(parts: list) -> str:
    payload = {
        "contents": [{
            
            "parts": parts
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 256
        }
    }

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            # Try to get error detail from Gemini response
            try:
                err_data = e.response.json()
                msg = err_data.get("error", {}).get("message", str(e))
            except:
                msg = str(e)
            raise HTTPException(status_code=e.response.status_code, detail=f"Gemini API error: {msg}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Failed to connect to Gemini: {e}")

    try:
        candidates = data.get("candidates", [])
        if not candidates:
            raise HTTPException(status_code=502, detail=f"Gemini returned no results: {json.dumps(data)}")
        
        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if not text:
            # Handle cases where safety filters might have blocked the response
            finish_reason = candidates[0].get("finishReason", "UNKNOWN")
            return f"[No text response. Reason: {finish_reason}]"

        return text.strip().replace("\n", " ")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"Error parsing Gemini response: {str(e)}. Data: {json.dumps(data)}")


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
