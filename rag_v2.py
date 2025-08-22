from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import uvicorn
import requests
from typing import Optional

from starlette.middleware.cors import CORSMiddleware

# Configuration de FastAPI
app = FastAPI(
    title="Assistant RAG",
    description="API pour télécharger des documents et poser des questions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration de Chromadb
chroma_client = chromadb.PersistentClient(path="chromadb-vdb")
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_vdb = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function,
)

# Configuration du modèle Ollama local
llm = Ollama(
    model="mistral:7b-instruct",
    base_url="http://localhost:11434",
    timeout=300,  # Augmentation du timeout à 5 minutes
    # Paramètres additionnels pour optimiser les performances
    num_predict=512,  # Limite la longueur de réponse
    temperature=0.7,
)

# Configuration du text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Template pour le prompt
prompt_template = """
Contexte: {context}

Question: {question}

Répondez à la question en utilisant uniquement le contexte fourni.
Si vous ne pouvez pas répondre à la question à partir du contexte, dites-le clairement.

Réponse:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
parser = StrOutputParser()
chain = prompt | llm | parser


# Modèles Pydantic pour les requêtes/réponses
class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    context_found: bool


class DocumentResponse(BaseModel):
    message: str
    document_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


def process_document(file_content: bytes, filename: str) -> str:
    """Traite un document (PDF ou Word) et retourne son contenu."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        temp_file.write(file_content)
        file_path = temp_file.name

    try:
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text = ' '.join([page.page_content for page in pages])
        elif filename.lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(file_path)
            text = loader.load()[0].page_content
        else:
            raise ValueError("Format de fichier non supporté. Formats acceptés: PDF, DOC, DOCX")

        return text
    finally:
        os.unlink(file_path)


def add_document_to_vectorstore(text: str, document_id: str):
    """Ajoute un document à la base de données vectorielle."""
    chunks = text_splitter.split_text(text)

    # Vérifier si le document existe déjà et le supprimer
    try:
        existing_ids = [f"{document_id}-chunk-{i}" for i in range(len(chunks) + 10)]  # Buffer pour être sûr
        chroma_vdb.delete(ids=existing_ids)
    except:
        pass  # Document n'existait pas

    # Ajout des nouveaux chunks
    chroma_vdb.add(
        documents=chunks,
        ids=[f"{document_id}-chunk-{i}" for i in range(len(chunks))],
        metadatas=[{"document_id": document_id} for _ in range(len(chunks))]
    )


def search_documents(query: str):
    """Recherche les documents pertinents."""
    results = chroma_vdb.query(
        query_texts=[query],
        n_results=3
    )
    return results


def check_ollama_connection():
    """Vérifie la connexion avec Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            mistral_available = any("mistral:7b-instruct" in model.get("name", "") for model in models)
            return mistral_available
        return False
    except:
        return False


@app.on_event("startup")
async def startup_event():
    """Vérifications au démarrage de l'application."""
    print("Vérification de la connexion Ollama...")
    if not check_ollama_connection():
        print("ATTENTION: Ollama n'est pas accessible ou le modèle mistral:7b-instruct n'est pas disponible")
        print("   Assurez-vous que:")
        print("   1. Ollama est démarré: ollama serve")
        print("   2. Le modèle est installé: ollama pull mistral:7b-instruct")
    else:
        print("Ollama est connecté et le modèle mistral:7b-instruct est disponible")


# Routes de l'API

@app.get("/", summary="Page d'accueil")
async def root():
    return {"message": "Assistant Documentaire RAG API", "version": "1.0.0"}


@app.get("/health", summary="Vérification de l'état de l'API")
async def health_check():
    ollama_status = check_ollama_connection()
    return {
        "status": "healthy",
        "service": "RAG API",
        "ollama_connected": ollama_status,
        "model": "mistral:7b-instruct"
    }


@app.get("/test-ollama", summary="Tester Ollama")
async def test_ollama():
    """Test de connexion et réponse d'Ollama."""
    try:
        # Test simple sans contexte
        test_response = llm.invoke("Dis bonjour en français.")
        return {
            "status": "success",
            "message": "Ollama fonctionne correctement",
            "test_response": test_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Ollama: {str(e)}")


@app.post("/upload-document",
          response_model=DocumentResponse,
          summary="Télécharger un document",
          description="Télécharge et traite un document PDF ou Word pour l'ajouter à la base de connaissances")
async def upload_document(file: UploadFile = File(...)):
    # Vérification du type de fichier
    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
        raise HTTPException(
            status_code=400,
            detail="Format de fichier non supporté. Formats acceptés: PDF, DOC, DOCX"
        )

    try:
        # Lecture du contenu du fichier
        file_content = await file.read()

        # Traitement du document
        text = process_document(file_content, file.filename)

        # Ajout à la base vectorielle
        add_document_to_vectorstore(text, file.filename)

        return DocumentResponse(
            message=f"Document '{file.filename}' ajouté avec succès!",
            document_id=file.filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du document: {str(e)}")


@app.post("/ask-question",
          response_model=QuestionResponse,
          summary="Poser une question",
          description="Pose une question basée sur les documents téléchargés")
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    try:
        # Recherche des documents pertinents
        results = search_documents(request.question)

        if not results["documents"] or not results["documents"][0]:
            return QuestionResponse(
                answer="Aucun document pertinent trouvé pour votre question. Assurez-vous d'avoir téléchargé des documents pertinents.",
                context_found=False
            )

        # Préparation du contexte
        context = "\n".join(results["documents"][0])

        # Génération de la réponse
        response = chain.invoke({
            "context": context,
            "question": request.question
        })

        return QuestionResponse(
            answer=response,
            context_found=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la réponse: {str(e)}")


@app.get("/documents", summary="Lister les documents")
async def list_documents():
    """Liste tous les documents dans la base de données."""
    try:
        # Récupérer tous les documents
        results = chroma_vdb.get()

        # Extraire les IDs uniques des documents
        document_ids = set()
        for metadata in results.get("metadatas", []):
            if metadata and "document_id" in metadata:
                document_ids.add(metadata["document_id"])

        return {
            "documents": list(document_ids),
            "total_chunks": len(results.get("ids", [])),
            "total_documents": len(document_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des documents: {str(e)}")


@app.delete("/documents/{document_id}", summary="Supprimer un document")
async def delete_document(document_id: str):
    """Supprime un document de la base de données."""
    try:
        # Récupérer tous les chunks du document
        results = chroma_vdb.get()
        ids_to_delete = []

        for i, metadata in enumerate(results.get("metadatas", [])):
            if metadata and metadata.get("document_id") == document_id:
                ids_to_delete.append(results["ids"][i])

        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' non trouvé")

        # Supprimer les chunks
        chroma_vdb.delete(ids=ids_to_delete)

        return {"message": f"Document '{document_id}' supprimé avec succès", "chunks_deleted": len(ids_to_delete)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
