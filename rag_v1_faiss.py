# rag_v1_faiss.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss
import os, pickle
from typing import Dict, List
from PyPDF2 import PdfReader
from docx import Document
import requests
import numpy as np
import json

app = FastAPI(title="RAG API - Mistral + CRUD docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Embeddings ===
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Persistance fichiers ===
FAISS_INDEX_FILE = "faiss_index.bin"
DOC_MAP_FILE = "doc_map.pkl"

# === Charger ou créer FAISS ===
dimension = embedding_model.get_sentence_embedding_dimension()
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOC_MAP_FILE):
    print("Chargement index FAISS existant...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(DOC_MAP_FILE, "rb") as f:
        doc_map = pickle.load(f)  # {chunk_id: {"text":..., "source":...}}
    doc_id_counter = max(doc_map.keys()) + 1 if doc_map else 0
else:
    print("Nouvel index FAISS...")
    index = faiss.IndexFlatL2(dimension)
    doc_map: Dict[int, Dict] = {}
    doc_id_counter = 0

# === Charger Mistral local ===
# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
#     load_in_8bit=True
# )
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


# === Mistral local via Ollama ===
def generate_with_ollama(prompt, model="mistral"):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt}
    response = requests.post(url, json=payload)
    response.raise_for_status()

    # return response.json()["response"]

    responses = response.text.strip().splitlines()
    last_response = json.loads(responses[-1])
    return last_response["response"]


# === Extraction texte ===
def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == ".docx":
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# === Sauvegarde état ===
def save_state():
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(DOC_MAP_FILE, "wb") as f:
        pickle.dump(doc_map, f)
    print("Sauvegarde effectuée")


# === Ingestion texte ===
def ingest_text(text: str, source: str):
    global doc_id_counter
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    embeddings = embedding_model.encode(chunks)

    for emb, chunk in zip(embeddings, chunks):
        # index.add([emb])
        index.add(np.array([emb]))
        doc_map[doc_id_counter] = {"text": chunk, "source": source}
        doc_id_counter += 1

    save_state()
    return len(chunks)


# === Upload fichier ===
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    text = file.file.read().decode("utf-8", errors="ignore")
    n_chunks = ingest_text(text, source=file.filename)
    return {"message": f"{file.filename} indexé avec succès", "chunks": n_chunks}


# === Ingestion dossier ===
@app.post("/ingest_folder")
def ingest_folder(folder_path: str):
    if not os.path.exists(folder_path):
        return {"error": "Dossier introuvable"}
    total_chunks = 0
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if os.path.isfile(path):
            text = extract_text(path)
            if text.strip():
                total_chunks += ingest_text(text, source=filename)
    return {"message": f"Ingestion terminée depuis {folder_path}", "chunks": total_chunks}


# === CRUD Documents ===
@app.get("/documents")
def list_documents():
    sources = {}
    for meta in doc_map.values():
        src = meta["source"]
        sources[src] = sources.get(src, 0) + 1
    return sources


@app.delete("/documents/{source_name}")
def delete_document(source_name: str):
    global doc_map, index
    # filtrer les chunks à garder
    keep_chunks = {i: meta for i, meta in doc_map.items() if meta["source"] != source_name}
    if len(keep_chunks) == len(doc_map):
        return {"error": f"Aucun document nommé {source_name} trouvé"}

    # recréer FAISS avec uniquement les chunks restants
    new_index = faiss.IndexFlatL2(dimension)
    new_map = {}
    new_id = 0
    texts = [meta["text"] for meta in keep_chunks.values()]
    embeddings = embedding_model.encode(texts)

    for emb, (old_id, meta) in zip(embeddings, keep_chunks.items()):
        new_index.add([emb])
        new_map[new_id] = meta
        new_id += 1

    # remplacer
    index = new_index
    doc_map = new_map
    save_state()

    return {"message": f"Document '{source_name}' supprimé avec succès"}


# === RAG Pipeline ===
# def rag_pipeline(question: str, k: int = 3) -> str:
#     q_embedding = embedding_model.encode([question])
#     distances, indices = index.search(q_embedding, k)
#     retrieved_docs = [doc_map[i]["text"] for i in indices[0] if i in doc_map]
#
#     context = "\n".join(retrieved_docs)
#     prompt = f"""
# Tu es un assistant. Utilise uniquement le contexte ci-dessous pour répondre.
#
# Contexte :
# {context}
#
# Question : {question}
# Réponse :
# """
#     # output = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.3)
#     # return output[0]["generated_text"]
#
#     output = generate_with_ollama(prompt)
#     return output

def rag_pipeline(question: str, k: int = 3) -> str:
    q_embedding = embedding_model.encode([question])
    distances, indices = index.search(q_embedding, k)
    retrieved_docs = [doc_map[i]["text"] for i in indices[0] if i in doc_map]

    context = "\n".join(retrieved_docs)
    prompt = f"""
Tu es un assistant. Utilise uniquement le contexte ci-dessous pour répondre.

Contexte :
{context}

Question : {question}
Réponse :
"""
    output = generate_with_ollama(prompt)
    return output


# === Question utilisateur ===
class Query(BaseModel):
    question: str
    k: int = 3


@app.post("/ask")
def ask_rag(query: Query):
    answer = rag_pipeline(query.question, query.k)
    return {"question": query.question, "answer": answer}
