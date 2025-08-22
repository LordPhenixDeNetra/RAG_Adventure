from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import chromadb
import uuid
import time
from datetime import datetime

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import uvicorn
import httpx
from typing import Optional, AsyncGenerator, Dict, Any
import json
import asyncio
from enum import Enum
from starlette.middleware.cors import CORSMiddleware

# Configuration de FastAPI
app = FastAPI(
    title="Assistant RAG Multi-Provider",
    description="API pour télécharger des documents et poser des questions avec support multi-provider",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Énumération des providers supportés
class Provider(str, Enum):
    MISTRAL = "mistral"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GROQ = "groq"

# Configuration des providers
PROVIDER_CONFIGS = {
    Provider.MISTRAL: {
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-medium",
        "headers_key": "Authorization",
        "headers_prefix": "Bearer"
    },
    Provider.OPENAI: {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-3.5-turbo",
        "headers_key": "Authorization",
        "headers_prefix": "Bearer"
    },
    Provider.ANTHROPIC: {
        "base_url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-haiku-20240307",
        "headers_key": "x-api-key",
        "headers_prefix": ""
    },
    Provider.DEEPSEEK: {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "headers_key": "Authorization",
        "headers_prefix": "Bearer"
    },
    Provider.GROQ: {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "mixtral-8x7b-32768",
        "headers_key": "Authorization",
        "headers_prefix": "Bearer"
    }
}

# Variables de configuration (à définir via variables d'environnement ou fichier config)
API_KEYS = {
    Provider.MISTRAL: os.getenv("MISTRAL_API_KEY", "VOTRE_CLE_MISTRAL_ICI"),  # Remplacez par votre vraie clé
    Provider.OPENAI: os.getenv("OPENAI_API_KEY", ""),
    Provider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY", ""),
    Provider.DEEPSEEK: os.getenv("DEEPSEEK_API_KEY", ""),
    Provider.GROQ: os.getenv("GROQ_API_KEY", "")
}

# Provider par défaut
DEFAULT_PROVIDER = Provider.MISTRAL

# Configuration de Chromadb
chroma_client = chromadb.PersistentClient(path="chromadb-vdb")
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_vdb = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function,
)

# Configuration du text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Template pour le prompt
prompt_template = """Contexte: {context}

Question: {question}

Répondez à la question en utilisant uniquement le contexte fourni.
Si vous ne pouvez pas répondre à la question à partir du contexte, dites-le clairement.

Réponse:"""

# Modèles Pydantic
class QuestionRequest(BaseModel):
    question: str
    provider: Optional[Provider] = DEFAULT_PROVIDER
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    enable_fallback: Optional[bool] = True  # Nouvelle option pour activer les fallbacks

class ProviderConfig(BaseModel):
    provider: Provider
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None

class QuestionResponse(BaseModel):
    id: str = Field(description="Identifiant unique de la réponse")
    answer: str
    context_found: bool
    provider_used: str
    model_used: str
    response_time_ms: float = Field(description="Temps de réponse en millisecondes")
    timestamp: str = Field(description="Date et heure de la réponse")

class DocumentResponse(BaseModel):
    id: str = Field(description="Identifiant unique de l'opération")
    message: str
    document_id: str
    timestamp: str = Field(description="Date et heure du téléchargement")

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class ProviderStatus(BaseModel):
    provider: str
    available: bool
    model: str
    has_api_key: bool

# Classes pour la gestion des providers
class LLMProvider:
    def __init__(self, provider: Provider):
        self.provider = provider
        self.config = PROVIDER_CONFIGS[provider]
        self.api_key = API_KEYS[provider]

    def get_headers(self) -> Dict[str, str]:
        """Retourne les headers pour l'API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            key = self.config["headers_key"]
            prefix = self.config["headers_prefix"]
            if prefix:
                headers[key] = f"{prefix} {self.api_key}"
            else:
                headers[key] = self.api_key
        return headers

    def format_messages(self, prompt: str) -> Dict[str, Any]:
        """Formate les messages selon le provider."""
        if self.provider == Provider.ANTHROPIC:
            return {
                "model": self.config["model"],
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            }
        else:
            return {
                "model": self.config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 512
            }

    def extract_response(self, response_data: Dict[str, Any]) -> str:
        """Extrait le texte de réponse selon le provider."""
        if self.provider == Provider.ANTHROPIC:
            return response_data["content"][0]["text"]
        else:
            return response_data["choices"][0]["message"]["content"]

    async def generate_response(self, prompt: str) -> str:
        """Génère une réponse via l'API."""
        if not self.api_key:
            raise ValueError(f"Clé API manquante pour {self.provider}")

        headers = self.get_headers()
        data = self.format_messages(prompt)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.config["base_url"],
                headers=headers,
                json=data
            )

            if response.status_code != 200:
                error_detail = response.text
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Erreur API {self.provider}: {error_detail}"
                )

            response_data = response.json()
            return self.extract_response(response_data)

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Génère une réponse en streaming."""
        if not self.api_key:
            raise ValueError(f"Clé API manquante pour {self.provider}")

        headers = self.get_headers()
        data = self.format_messages(prompt)
        data["stream"] = True

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                    'POST',
                    self.config["base_url"],
                    headers=headers,
                    json=data
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Erreur API {self.provider}: {response.text}"
                    )

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if self.provider == Provider.ANTHROPIC:
                                if data.get("type") == "content_block_delta":
                                    yield data["delta"]["text"]
                            else:
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                        except json.JSONDecodeError:
                            continue

# Fonctions utilitaires
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

    # Supprimer l'ancien document s'il existe
    try:
        existing_ids = [f"{document_id}-chunk-{i}" for i in range(len(chunks) + 10)]
        chroma_vdb.delete(ids=existing_ids)
    except:
        pass

    # Ajouter les nouveaux chunks
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

async def check_provider_status(provider: Provider) -> ProviderStatus:
    """Vérifie le statut d'un provider."""
    config = PROVIDER_CONFIGS[provider]
    api_key = API_KEYS[provider]

    return ProviderStatus(
        provider=provider.value,
        available=bool(api_key),
        model=config["model"],
        has_api_key=bool(api_key)
    )

# Routes de l'API
@app.get("/", summary="Page d'accueil")
async def root():
    return {
        "message": "Assistant Documentaire RAG API Multi-Provider",
        "version": "2.0.0",
        "supported_providers": [p.value for p in Provider]
    }

@app.get("/health", summary="Vérification de l'état de l'API")
async def health_check():
    provider_statuses = {}
    for provider in Provider:
        status = await check_provider_status(provider)
        provider_statuses[provider.value] = status.dict()

    return {
        "status": "healthy",
        "service": "RAG API Multi-Provider",
        "providers": provider_statuses
    }

@app.get("/providers", summary="Lister les providers disponibles")
async def list_providers():
    """Liste tous les providers et leur statut."""
    providers = []
    for provider in Provider:
        status = await check_provider_status(provider)
        providers.append(status.dict())

    return {"providers": providers}

@app.post("/configure-provider", summary="Configurer un provider")
async def configure_provider(config: ProviderConfig):
    """Configure un provider avec une nouvelle clé API."""
    if config.api_key:
        API_KEYS[config.provider] = config.api_key

    if config.model:
        PROVIDER_CONFIGS[config.provider]["model"] = config.model

    if config.base_url:
        PROVIDER_CONFIGS[config.provider]["base_url"] = config.base_url

    return {"message": f"Provider {config.provider} configuré avec succès"}

@app.post("/upload-document",
          response_model=DocumentResponse,
          summary="Télécharger un document")
async def upload_document(file: UploadFile = File(...)):
    # Génération de l'ID unique et timestamp
    operation_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
        raise HTTPException(
            status_code=400,
            detail="Format de fichier non supporté. Formats acceptés: PDF, DOC, DOCX"
        )

    try:
        file_content = await file.read()
        text = process_document(file_content, file.filename)
        add_document_to_vectorstore(text, file.filename)

        return DocumentResponse(
            id=operation_id,
            message=f"Document '{file.filename}' ajouté avec succès!",
            document_id=file.filename,
            timestamp=timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du document: {str(e)}")

@app.post("/ask-question",
          response_model=QuestionResponse,
          summary="Poser une question")
async def ask_question(request: QuestionRequest):
    # Génération de l'ID unique et timestamp de début
    response_id = str(uuid.uuid4())
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    try:
        # Recherche des documents pertinents
        results = search_documents(request.question)

        if not results["documents"] or not results["documents"][0]:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            return QuestionResponse(
                id=response_id,
                answer="Aucun document pertinent trouvé pour votre question.",
                context_found=False,
                provider_used=request.provider.value,
                model_used=PROVIDER_CONFIGS[request.provider]["model"],
                response_time_ms=round(response_time_ms, 2),
                timestamp=timestamp
            )

        # Préparation du contexte
        context = "\n".join(results["documents"][0])
        prompt = prompt_template.format(context=context, question=request.question)

        # Génération de la réponse avec fallbacks
        try:
            response, provider_used, model_used = await try_providers_with_fallback(
                prompt,
                request.provider,
                request.model,
                request.enable_fallback
            )
        except HTTPException as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            raise e

        # Calcul du temps de réponse
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        return QuestionResponse(
            id=response_id,
            answer=response,
            context_found=True,
            provider_used=provider_used,
            model_used=model_used,
            response_time_ms=round(response_time_ms, 2),
            timestamp=timestamp
        )

    except Exception as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la réponse: {str(e)}")

@app.post("/ask-question-stream",
          summary="Poser une question (streaming)")
async def ask_question_stream(request: QuestionRequest):
    # Génération de l'ID unique et timestamp
    response_id = str(uuid.uuid4())
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    try:
        # Recherche des documents pertinents
        results = search_documents(request.question)

        if not results["documents"] or not results["documents"][0]:
            async def no_context_stream():
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000

                message = "Aucun document pertinent trouvé pour votre question."
                metadata = {
                    "id": response_id,
                    "provider_used": request.provider.value,
                    "model_used": PROVIDER_CONFIGS[request.provider]["model"],
                    "response_time_ms": round(response_time_ms, 2),
                    "timestamp": timestamp,
                    "context_found": False
                }
                yield f"data: {json.dumps({'content': message, 'done': True, 'metadata': metadata})}\n\n"

            return StreamingResponse(
                no_context_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )

        # Préparation du contexte
        context = "\n".join(results["documents"][0])
        prompt = prompt_template.format(context=context, question=request.question)

        # Génération de la réponse en streaming
        provider = LLMProvider(request.provider)

        if request.model:
            provider.config["model"] = request.model

        async def generate_stream():
            try:
                # Envoyer d'abord les métadonnées
                initial_metadata = {
                    "id": response_id,
                    "provider_used": request.provider.value,
                    "model_used": provider.config["model"],
                    "timestamp": timestamp,
                    "context_found": True
                }
                yield f"data: {json.dumps({'content': '', 'done': False, 'metadata': initial_metadata})}\n\n"

                async for chunk in provider.generate_stream(prompt):
                    if chunk:
                        yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"

                # Signal de fin avec temps de réponse final
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                final_metadata = {
                    "response_time_ms": round(response_time_ms, 2)
                }
                yield f"data: {json.dumps({'content': '', 'done': True, 'metadata': final_metadata})}\n\n"

            except Exception as e:
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                error_metadata = {
                    "response_time_ms": round(response_time_ms, 2)
                }
                yield f"data: {json.dumps({'error': f'Erreur: {str(e)}', 'metadata': error_metadata})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except Exception as e:
        async def error_stream():
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            error_metadata = {
                "id": response_id,
                "response_time_ms": round(response_time_ms, 2),
                "timestamp": timestamp
            }
            yield f"data: {json.dumps({'error': f'Erreur: {str(e)}', 'metadata': error_metadata})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/plain")

@app.get("/documents", summary="Lister les documents")
async def list_documents():
    """Liste tous les documents dans la base de données."""
    try:
        results = chroma_vdb.get()
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
        results = chroma_vdb.get()
        ids_to_delete = []

        for i, metadata in enumerate(results.get("metadatas", [])):
            if metadata and metadata.get("document_id") == document_id:
                ids_to_delete.append(results["ids"][i])

        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' non trouvé")

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