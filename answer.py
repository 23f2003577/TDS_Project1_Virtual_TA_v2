# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "argparse",
#     "fastapi",
#     "httpx",
#     "markdownify",
#     "numpy",
#     "semantic-text-splitter",
#     "tqdm",
#     "uvicorn",
#     "sentence-transformers",
#     "pillow",
#     "html2text",
#     "beautifulsoup4",
#     "python-dotenv",
# ]
# ///

import os
import time
import json
import base64
import httpx
import numpy as np
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_embeddings():
    data = np.load("still_merged_embeddings.npz", allow_pickle=True)
    return data["chunks"], data["embeddings"]

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
                    "content": "You are a helpful teaching assistant that answers questions based on provided context. Use the provided context to answer the question. Your response should be in markdown format. If the context is insufficient, respond with 'I don't know'."
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

def answer(question: str, image: str = None):
    chunks, embeddings = load_embeddings()

    if image:
        question += " (image received, but image captioning is not supported in local version)"

    q_embedding = model.encode(question)
    similarities = np.dot(embeddings, q_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embedding)
    )

    top_indices = np.argsort(similarities)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_indices]

    response = generate_llm_response(question, "\n".join(top_chunks))
    return {
        "question": question,
        "response": response,
        "top_chunks": top_chunks
    }

@app.post("/api/")
async def api_answer(request: Request):
    try:
        data = await request.json()
        return answer(data.get('question'), data.get('image'))
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Welcome to the Q&A API. Use POST /api/ with JSON payload {'question': 'your question', 'image': 'base64 image data'} to get answers."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
