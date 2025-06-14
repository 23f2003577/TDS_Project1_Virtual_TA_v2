# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi",
#     "httpx",
#     "numpy",
#     "sentence-transformers==2.2.2",
#     "python-dotenv"
# ]
# ///

import os
import httpx
import numpy as np
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Load model only once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load your pre-generated embeddings
def load_embeddings():
    data = np.load("still_merged_embeddings.npz", allow_pickle=True)
    return data["chunks"], data["embeddings"]

# Call LLM (Mistral via AIPipe)
def generate_llm_response(question: str, context: str):
    OPENROUTER_TOKEN = os.getenv("AIPIPE_TOKEN")
    response = httpx.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful teaching assistant that answers questions based on provided context. Use the context to answer in markdown. Say 'I don't know' if unsure."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}"
                }
            ],
            "temperature": 0.5,
            "max_tokens": 512
        },
        timeout=30.0
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Main answer function
def answer(question: str, image: str = None):
    chunks, embeddings = load_embeddings()

    # Append note about image if present
    if image:
        question += " (image received, but image captioning is not supported in local version)"

    # Embed question and compute similarities
    q_embedding = model.encode(question)
    similarities = np.dot(embeddings, q_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embedding)
    )

    top_indices = np.argsort(similarities)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    context = "\n".join(top_chunks)

    # Call the LLM for the answer
    response = generate_llm_response(question, context)

    # Extract discourse links from the context
    links = []
    for chunk in top_chunks:
        for word in chunk.split():
            if "discourse.onlinedegree.iitm.ac.in" in word:
                url = word.strip("()<>.,\"'")
                if url not in [l["url"] for l in links]:
                    links.append({
                        "url": url,
                        "text": "Discourse Link"
                    })

    return {
        "answer": response,
        "links": links
    }

# FastAPI route
@app.post("/api/")
async def api_answer(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        image = data.get("image")  # Optional
        return answer(question, image)
    except Exception as e:
        return {"error": str(e)}

# Welcome route (optional)
@app.get("/")
async def root():
    return {"message": "Welcome to the Q&A API. Use POST /api/ with {'question': '...', 'image': '...'} to get answers."}

# Local dev entrypoint (not used on Vercel)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
