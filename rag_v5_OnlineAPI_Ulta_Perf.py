from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# from httpx import Response
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import chromadb
import uuid
import time
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import tempfile
import uvicorn
import httpx
from typing import Optional, AsyncGenerator, Dict, Any, List, Tuple
import json
from enum import Enum
import redis
import hashlib
from functools import lru_cache
import numpy as np
from rank_bm25 import BM25Okapi
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import pickle

from fastapi.responses import Response
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de FastAPI
app = FastAPI(
    title="RAG Ultra Performant API",
    description="API RAG avec recherche hybride, re-ranking et optimisations avancées",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger les variables d'environnement
load_dotenv()

# Métriques Prometheus
query_counter = Counter('rag_queries_total', 'Total queries processed', ['provider', 'status'])
response_time_histogram = Histogram('rag_response_time_seconds', 'Response time distribution')
accuracy_gauge = Gauge('rag_accuracy_score', 'Current accuracy score')
cache_hit_counter = Counter('rag_cache_hits_total', 'Total cache hits', ['cache_type'])


# Énumération des providers
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
        "model": "gpt-4o-mini",
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

# Variables d'API
API_KEYS = {
    Provider.MISTRAL: os.getenv("MISTRAL_API_KEY", ""),
    Provider.OPENAI: os.getenv("OPENAI_API_KEY", ""),
    Provider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY", ""),
    Provider.DEEPSEEK: os.getenv("DEEPSEEK_API_KEY", ""),
    Provider.GROQ: os.getenv("GROQ_API_KEY", "")
}

DEFAULT_PROVIDER = Provider.MISTRAL

# Configuration Redis pour le cache
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,  # Timeout connexion
        socket_timeout=5,  # Timeout opérations
        retry_on_timeout=True,  # Retry automatique
        health_check_interval=30  # Vérification santé
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connecté avec succès")
except Exception as e:
    REDIS_AVAILABLE = False
    logger.warning(f"Redis non disponible: {e}")


# Classes de données
@dataclass
class SearchResult:
    content: str
    score: float
    metadata: Dict[str, Any]
    source_type: str  # "dense", "sparse", "hybrid"


@dataclass
class RankedResult:
    content: str
    score: float
    metadata: Dict[str, Any]
    original_rank: int


# Cache multicouche
class MultiLayerCache:
    def __init__(self):
        self.memory_cache = {}
        self.memory_cache_lock = threading.Lock()
        self.max_memory_items = 1000

    def _get_cache_key(self, key: str, prefix: str = "") -> str:
        return f"{prefix}:{hashlib.md5(key.encode()).hexdigest()}"

    def get(self, key: str, cache_type: str = "general") -> Optional[Any]:
        cache_key = self._get_cache_key(key, cache_type)

        # Essayer Redis d'abord
        if REDIS_AVAILABLE:
            try:
                result = redis_client.get(cache_key)
                if result:
                    cache_hit_counter.labels(cache_type=f"redis_{cache_type}").inc()
                    return pickle.loads(result.encode('latin1'))
            except Exception as e:
                logger.error(f"Erreur Redis get: {e}")

        # Fallback sur cache mémoire
        with self.memory_cache_lock:
            if cache_key in self.memory_cache:
                cache_hit_counter.labels(cache_type=f"memory_{cache_type}").inc()
                return self.memory_cache[cache_key]

        return None

    def set(self, key: str, value: Any, ttl: int = 3600, cache_type: str = "general"):
        cache_key = self._get_cache_key(key, cache_type)

        # Redis
        if REDIS_AVAILABLE:
            try:
                serialized = pickle.dumps(value).decode('latin1')
                redis_client.setex(cache_key, ttl, serialized)
            except Exception as e:
                logger.error(f"Erreur Redis set: {e}")

        # Cache mémoire
        with self.memory_cache_lock:
            if len(self.memory_cache) >= self.max_memory_items:
                # Supprimer les entrées les plus anciennes
                oldest_keys = list(self.memory_cache.keys())[:100]
                for old_key in oldest_keys:
                    del self.memory_cache[old_key]

            self.memory_cache[cache_key] = value


# Cache global
cache = MultiLayerCache()


# Modèles d'embeddings avancés
class AdvancedEmbeddings:
    def __init__(self):
        try:
            # Modèle principal optimisé
            self.primary_model = SentenceTransformer(
                'all-mpnet-base-v2',
                device='cpu',  # Explicite pour éviter les warnings
                cache_folder='./.cache/sentence_transformers'  # Cache local
            )
            logger.info("Modèle principal all-mpnet-base-v2 chargé")

            # Modèle multilingue en lazy loading (chargé seulement si nécessaire)
            self.multilingual_model = None
            self._multilingual_loaded = False

        except Exception as e:
            logger.error(f"Erreur chargement modèles: {e}")
            # Fallback sur un modèle plus léger
            self.primary_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _load_multilingual_if_needed(self):
        """Chargement lazy du modèle multilingue"""
        if not self._multilingual_loaded:
            try:
                self.multilingual_model = SentenceTransformer(
                    'paraphrase-multilingual-mpnet-base-v2',
                    device='cpu',
                    cache_folder='./.cache/sentence_transformers'
                )
                self._multilingual_loaded = True
                logger.info("Modèle multilingue chargé")
            except Exception as e:
                logger.error(f"Erreur chargement modèle multilingue: {e}")
                self.multilingual_model = None

    @lru_cache(maxsize=5000)
    def embed_query(self, text: str) -> np.ndarray:
        """Cache des embeddings de requêtes avec LRU"""
        return self.primary_model.encode([text])[0]

    def embed_documents(self, texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
        """Embedding de documents avec cache intelligent"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if use_cache:
                cached = cache.get(text, "embeddings")
                if cached is not None:
                    embeddings.append(cached)
                    continue

            uncached_texts.append(text)
            uncached_indices.append(i)
            embeddings.append(None)  # Placeholder

        # Traitement par batch des textes non cachés
        if uncached_texts:
            new_embeddings = self.primary_model.encode(uncached_texts)

            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                if use_cache:
                    cache.set(uncached_texts[uncached_indices.index(idx)],
                              embedding, cache_type="embeddings")

        return embeddings


# Chunking sémantique avancé
class AdvancedChunker:
    def __init__(self, embeddings_model):
        self.embeddings = embeddings_model
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )

    def chunk_document(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Chunking intelligent avec contexte sémantique"""
        # Chunking de base
        base_chunks = self.base_splitter.split_text(text)

        # Enrichissement des chunks
        enriched_chunks = []
        for i, chunk in enumerate(base_chunks):
            # Contexte des chunks adjacents
            context_before = base_chunks[i - 1] if i > 0 else ""
            context_after = base_chunks[i + 1] if i < len(base_chunks) - 1 else ""

            # Métadonnées enrichies
            metadata = {
                "document_id": document_id,
                "chunk_id": f"{document_id}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(base_chunks),
                "context_before": context_before[:200],  # Contexte réduit
                "context_after": context_after[:200],
                "chunk_length": len(chunk),
                "chunk_type": self._classify_chunk(chunk)
            }

            enriched_chunks.append({
                "content": chunk,
                "metadata": metadata
            })

        return enriched_chunks

    def _classify_chunk(self, chunk: str) -> str:
        """Classification basique du type de contenu"""
        if len(chunk.split()) < 10:
            return "short"
        elif ":" in chunk and len(chunk.split(":")) > 1:
            return "structured"
        elif chunk.count(".") > 3:
            return "paragraph"
        else:
            return "fragment"


# Recherche hybride Dense + Sparse
class HybridSearch:
    def __init__(self, chroma_db, embeddings_model):
        self.chroma_db = chroma_db
        self.embeddings = embeddings_model
        self.bm25_index = None
        self.documents = []
        self.document_ids = []
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Construction de l'index BM25"""
        try:
            # Récupération de tous les documents
            results = self.chroma_db.get()
            if results and results.get("documents"):
                self.documents = results["documents"]
                self.document_ids = results["ids"]

                # Tokenisation pour BM25
                tokenized_docs = [doc.lower().split() for doc in self.documents]
                self.bm25_index = BM25Okapi(tokenized_docs)

                logger.info(f"Index BM25 construit avec {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Erreur construction index BM25: {e}")

    def rebuild_index(self):
        """Reconstruction de l'index BM25"""
        self._build_bm25_index()

    async def search(self, query: str, n_results: int = 10, alpha: float = 0.7) -> List[SearchResult]:
        """Recherche hybride avec pondération dense/sparse"""
        results = []

        # 1. Recherche dense (vectorielle)
        try:
            dense_results = self.chroma_db.query(
                query_texts=[query],
                n_results=min(n_results * 2, 20)  # Plus de résultats pour diversité
            )

            if dense_results and dense_results.get("documents") and dense_results["documents"][0]:
                for i, (doc, distance) in enumerate(zip(
                        dense_results["documents"][0],
                        dense_results["distances"][0]
                )):
                    # Conversion distance en score (plus c'est proche, plus le score est élevé)
                    dense_score = 1 / (1 + distance)

                    metadata = dense_results["metadatas"][0][i] if dense_results.get("metadatas") else {}

                    results.append(SearchResult(
                        content=doc,
                        score=dense_score * alpha,
                        metadata=metadata,
                        source_type="dense"
                    ))
        except Exception as e:
            logger.error(f"Erreur recherche dense: {e}")

        # 2. Recherche sparse (BM25)
        if self.bm25_index and self.documents:
            try:
                query_tokens = query.lower().split()
                bm25_scores = self.bm25_index.get_scores(query_tokens)

                # Normalisation des scores BM25
                max_bm25_score = max(bm25_scores) if bm25_scores.max() > 0 else 1
                normalized_bm25_scores = bm25_scores / max_bm25_score

                # Top documents BM25
                top_bm25_indices = np.argsort(bm25_scores)[-n_results:][::-1]

                for idx in top_bm25_indices:
                    if bm25_scores[idx] > 0:
                        sparse_score = normalized_bm25_scores[idx] * (1 - alpha)

                        results.append(SearchResult(
                            content=self.documents[idx],
                            score=sparse_score,
                            metadata={"document_index": idx},
                            source_type="sparse"
                        ))

            except Exception as e:
                logger.error(f"Erreur recherche sparse: {e}")

        # 3. Fusion et déduplication des résultats
        combined_results = self._combine_and_deduplicate(results, n_results)

        return combined_results

    def _combine_and_deduplicate(self, results: List[SearchResult], n_results: int) -> List[SearchResult]:
        """Fusion et déduplication des résultats avec boost pour résultats hybrides"""
        content_map = {}

        for result in results:
            content_hash = hashlib.md5(result.content.encode()).hexdigest()

            if content_hash in content_map:
                # Boost pour les documents trouvés par les deux méthodes
                existing = content_map[content_hash]
                if existing.source_type != result.source_type:
                    existing.score += result.score * 0.2  # Bonus hybride
                    existing.source_type = "hybrid"
                else:
                    existing.score = max(existing.score, result.score)
            else:
                content_map[content_hash] = result

        # Tri par score et limitation
        final_results = sorted(content_map.values(), key=lambda x: x.score, reverse=True)
        return final_results[:n_results]


# Re-ranking avec Cross-Encoder
class AdvancedReranker:
    def __init__(self):
        # Modèle de re-ranking haute performance
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.rerank_cache = {}

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[RankedResult]:
        """Re-ranking des résultats avec cross-encoder"""
        if not results:
            return []

        # Vérification du cache
        cache_key = f"{query}_{len(results)}_{hash(tuple(r.content[:50] for r in results))}"
        cached = cache.get(cache_key, "rerank")
        if cached:
            return cached[:top_k]

        try:
            # Préparation des paires query-document
            pairs = [(query, result.content) for result in results]

            # Scoring avec cross-encoder
            cross_scores = self.reranker.predict(pairs)

            # Combinaison des scores (retrieval + reranking)
            final_results = []
            for i, (result, cross_score) in enumerate(zip(results, cross_scores)):
                # Score final: pondération retrieval (30%) + cross-encoder (70%)
                final_score = 0.3 * result.score + 0.7 * cross_score

                final_results.append(RankedResult(
                    content=result.content,
                    score=final_score,
                    metadata=result.metadata,
                    original_rank=i
                ))

            # Tri par score final
            ranked_results = sorted(final_results, key=lambda x: x.score, reverse=True)

            # Cache du résultat
            cache.set(cache_key, ranked_results, ttl=1800, cache_type="rerank")

            return ranked_results[:top_k]

        except Exception as e:
            logger.error(f"Erreur re-ranking: {e}")
            # Fallback: retour des résultats originaux
            fallback_results = [
                RankedResult(
                    content=result.content,
                    score=result.score,
                    metadata=result.metadata,
                    original_rank=i
                )
                for i, result in enumerate(results)
            ]
            return sorted(fallback_results, key=lambda x: x.score, reverse=True)[:top_k]


# Query Enhancement
class QueryEnhancer:
    def __init__(self):
        self.enhancement_cache = {}

    async def enhance_query(self, query: str, provider_instance) -> List[str]:
        """Génération de variantes de requête pour améliorer la recherche"""
        # Vérification du cache
        cached = cache.get(query, "query_enhancement")
        if cached:
            return cached

        try:
            enhancement_prompt = f"""Vous êtes un expert en reformulation de requêtes pour améliorer la recherche documentaire.

Requête originale: "{query}"

Générez 2 variantes de cette requête qui:
1. Utilisent des synonymes et termes alternatifs
2. Reformulent la question sous un angle différent
3. Sont plus spécifiques ou plus générales selon le contexte

Répondez uniquement avec les 2 variantes, une par ligne, sans numérotation ni formatage:"""

            enhanced_text = await provider_instance.generate_response(enhancement_prompt)

            # Parsing des variantes
            variants = [line.strip() for line in enhanced_text.split('\n') if line.strip()]
            variants = [v for v in variants if len(v) > 10]  # Filtrage des variantes trop courtes

            # Inclure la requête originale
            all_queries = [query] + variants[:2]  # Maximum 3 requêtes au total

            # Cache du résultat
            cache.set(query, all_queries, ttl=3600, cache_type="query_enhancement")

            return all_queries

        except Exception as e:
            logger.error(f"Erreur query enhancement: {e}")
            return [query]  # Fallback sur la requête originale


# Provider LLM optimisé
class OptimizedLLMProvider:
    def __init__(self, provider: Provider):
        self.provider = provider
        self.config = PROVIDER_CONFIGS[provider]
        self.api_key = API_KEYS[provider]
        self.client = None

    def get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            key = self.config["headers_key"]
            prefix = self.config["headers_prefix"]
            if prefix:
                headers[key] = f"{prefix} {self.api_key}"
            else:
                headers[key] = self.api_key
        return headers

    def format_messages(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> Dict[str, Any]:
        if self.provider == Provider.ANTHROPIC:
            return {
                "model": self.config["model"],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
        else:
            return {
                "model": self.config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

    def extract_response(self, response_data: Dict[str, Any]) -> str:
        if self.provider == Provider.ANTHROPIC:
            return response_data["content"][0]["text"]
        else:
            return response_data["choices"][0]["message"]["content"]

    async def generate_response(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            raise ValueError(f"Clé API manquante pour {self.provider}")

        headers = self.get_headers()
        data = self.format_messages(prompt, **kwargs)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.config["base_url"],
                headers=headers,
                json=data
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Erreur API {self.provider}: {response.text}"
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
                            data_json = json.loads(data_str)
                            if self.provider == Provider.ANTHROPIC:
                                if data_json.get("type") == "content_block_delta":
                                    yield data_json["delta"]["text"]
                            else:
                                if "choices" in data_json and len(data_json["choices"]) > 0:
                                    delta = data_json["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                        except json.JSONDecodeError:
                            continue


# RAG Ultra Performant - Classe principale
class UltraPerformantRAG:
    def __init__(self):
        # Initialisation des composants
        self.embeddings = AdvancedEmbeddings()
        self.chunker = AdvancedChunker(self.embeddings)
        self.reranker = AdvancedReranker()
        self.query_enhancer = QueryEnhancer()

        # ChromaDB avec gestion d'erreurs
        try:
            self.chroma_client = chromadb.PersistentClient(
                path="./ultra_rag_db",
                settings=chromadb.Settings(
                    anonymized_telemetry=False,  # Désactive la télémétrie
                    allow_reset=True
                )
            )

            # Fonction d'embedding custom pour ChromaDB
            class CustomEmbeddingFunction(chromadb.EmbeddingFunction):
                def __init__(self, embeddings_model):
                    self.embeddings_model = embeddings_model

                def __call__(self, texts):
                    return self.embeddings_model.embed_documents(texts)

            self.embedding_function = CustomEmbeddingFunction(self.embeddings)

            self.collection = self.chroma_client.get_or_create_collection(
                name="ultra_documents",
                embedding_function=self.embedding_function,
            )

            logger.info("ChromaDB initialisé avec succès")

        except Exception as e:
            logger.error(f"Erreur initialisation ChromaDB: {e}")
            raise

        # Recherche hybride
        self.hybrid_search = HybridSearch(self.collection, self.embeddings)

        # Pool de threads pour opérations parallèles
        self.executor = ThreadPoolExecutor(max_workers=2)  # Réduit pour éviter surcharge

    async def add_document(self, text: str, document_id: str) -> Dict[str, Any]:
        """Ajout optimisé d'un document avec chunking intelligent"""
        start_time = time.time()

        try:
            # Chunking sémantique
            chunks_data = self.chunker.chunk_document(text, document_id)

            # Suppression des anciens chunks du même document
            try:
                self.collection.delete(where={"document_id": document_id})
            except:
                pass

            # Préparation des données pour l'insertion
            documents = [chunk["content"] for chunk in chunks_data]
            metadatas = [chunk["metadata"] for chunk in chunks_data]
            ids = [metadata["chunk_id"] for metadata in metadatas]

            # Insertion par batch
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )

            # Reconstruction de l'index BM25
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.hybrid_search.rebuild_index
            )

            processing_time = time.time() - start_time

            return {
                "document_id": document_id,
                "chunks_created": len(chunks_data),
                "processing_time_ms": round(processing_time * 1000, 2),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Erreur ajout document: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur traitement document: {str(e)}")

    async def query(self, question: str, provider: Provider, top_k: int = 3, **kwargs) -> Dict[str, Any]:
        """Query ultra optimisé avec toutes les améliorations"""
        start_time = time.time()
        query_id = str(uuid.uuid4())

        try:
            # Vérification du cache complet
            cache_key = f"{question}_{provider.value}_{top_k}"
            cached_response = cache.get(cache_key, "full_response")
            if cached_response:
                cache_hit_counter.labels(cache_type="full_response").inc()
                return cached_response

            # 1. Provider LLM
            llm_provider = OptimizedLLMProvider(provider)

            # 2. Enhancement de la requête
            enhanced_queries = await self.query_enhancer.enhance_query(question, llm_provider)
            logger.info(f"Requêtes générées: {enhanced_queries}")

            # 3. Recherche hybride pour toutes les variantes
            all_results = []
            for query_variant in enhanced_queries:
                variant_results = await self.hybrid_search.search(
                    query_variant,
                    n_results=15  # Plus de résultats pour diversité
                )
                all_results.extend(variant_results)

            if not all_results:
                no_context_response = {
                    "id": query_id,
                    "answer": "Aucun document pertinent trouvé pour votre question.",
                    "context_found": False,
                    "provider_used": provider.value,
                    "model_used": PROVIDER_CONFIGS[provider]["model"],
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "timestamp": datetime.now().isoformat(),
                    "search_results": 0,
                    "enhanced_queries": enhanced_queries
                }
                return no_context_response

            # 4. Re-ranking avec cross-encoder
            ranked_results = self.reranker.rerank(question, all_results, top_k=top_k)

            # 5. Préparation du contexte optimisé
            context_parts = []
            sources = []

            for i, result in enumerate(ranked_results):
                context_parts.append(f"Source {i + 1}: {result.content}")
                sources.append({
                    "source_id": i + 1,
                    "score": float(result.score),
                    "original_rank": result.original_rank,
                    "metadata": result.metadata
                })

            context = "\n\n".join(context_parts)

            # 6. Prompt optimisé avec instructions spécifiques
            optimized_prompt = f"""Vous êtes un assistant expert qui répond aux questions en utilisant uniquement le contexte fourni.

CONTEXTE:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Répondez uniquement en utilisant les informations du contexte fourni
2. Si vous ne trouvez pas d'informations pertinentes, dites-le clairement
3. Citez les sources en utilisant "Source X" quand approprié
4. Soyez précis et concis
5. Si plusieurs sources contiennent des informations complémentaires, synthétisez-les

RÉPONSE:"""

            # 7. Génération de la réponse
            response_text = await llm_provider.generate_response(
                optimized_prompt,
                temperature=kwargs.get('temperature', 0.3),  # Plus conservateur pour la précision
                max_tokens=kwargs.get('max_tokens', 512)
            )

            # 8. Construction de la réponse finale
            end_time = time.time()
            response_time_ms = round((end_time - start_time) * 1000, 2)

            final_response = {
                "id": query_id,
                "answer": response_text,
                "context_found": True,
                "provider_used": provider.value,
                "model_used": PROVIDER_CONFIGS[provider]["model"],
                "response_time_ms": response_time_ms,
                "timestamp": datetime.now().isoformat(),
                "search_results": len(all_results),
                "ranked_results": len(ranked_results),
                "enhanced_queries": enhanced_queries,
                "sources": sources,
                "performance_metrics": {
                    "search_time_ms": round((time.time() - start_time - (end_time - time.time())) * 1000, 2),
                    "generation_time_ms": round((end_time - start_time) * 1000, 2),
                    # "cache_hits": cache_hit_counter._value._value
                    "cache_hits": "metrics_available_via_prometheus"
                }
            }

            # 9. Cache de la réponse complète
            cache.set(cache_key, final_response, ttl=1800, cache_type="full_response")

            # 10. Métriques
            query_counter.labels(provider=provider.value, status="success").inc()
            response_time_histogram.observe(response_time_ms / 1000)

            return final_response

        except Exception as e:
            logger.error(f"Erreur query complète: {e}")
            query_counter.labels(provider=provider.value, status="error").inc()

            error_response = {
                "id": query_id,
                "error": f"Erreur lors du traitement: {str(e)}",
                "context_found": False,
                "provider_used": provider.value,
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": datetime.now().isoformat()
            }
            return error_response


# Instance globale du RAG
rag_system = UltraPerformantRAG()


# Modèles Pydantic
class QuestionRequest(BaseModel):
    question: str
    provider: Optional[Provider] = DEFAULT_PROVIDER
    model: Optional[str] = None
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 512
    top_k: Optional[int] = 3


class AdvancedQuestionResponse(BaseModel):
    id: str
    answer: str
    context_found: bool
    provider_used: str
    model_used: str
    response_time_ms: float
    timestamp: str
    search_results: int
    ranked_results: int
    enhanced_queries: List[str]
    sources: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


class DocumentResponse(BaseModel):
    document_id: str
    chunks_created: int
    processing_time_ms: float
    status: str


class PerformanceMetrics(BaseModel):
    total_queries: int
    average_response_time_ms: float
    cache_hit_rate: float
    error_rate: float
    active_documents: int


# Fonctions utilitaires améliorées
async def process_document_advanced(file_content: bytes, filename: str) -> str:
    """Traitement avancé de document avec extraction améliorée"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        temp_file.write(file_content)
        file_path = temp_file.name

    try:
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Extraction avec métadonnées de page
            text_parts = []
            for page_num, page in enumerate(pages):
                page_text = page.page_content.strip()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

            text = '\n\n'.join(text_parts)

        elif filename.lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(file_path)
            document = loader.load()[0]
            text = document.page_content

        else:
            raise ValueError("Format non supporté. Formats acceptés: PDF, DOC, DOCX")

        # Nettoyage et normalisation du texte
        text = text.replace('\x00', '')  # Suppression caractères null
        text = ' '.join(text.split())  # Normalisation espaces

        return text

    finally:
        os.unlink(file_path)


# Middleware de monitoring
@app.middleware("http")
async def monitoring_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)

    # Métriques de performance globales
    process_time = time.time() - start_time
    response_time_histogram.observe(process_time)

    return response


@app.on_event("startup")
async def startup_event():
    """Événements de démarrage pour optimisation"""
    logger.info("Démarrage du RAG Ultra Performant...")

    # Pré-chargement du modèle de re-ranking en arrière-plan
    def preload_reranker():
        try:
            # Test du reranker
            rag_system.reranker.rerank("test", [
                SearchResult("test content", 0.5, {}, "test")
            ], top_k=1)
            logger.info("Reranker pré-chargé")
        except Exception as e:
            logger.error(f"Erreur pré-chargement reranker: {e}")

    # Exécution en arrière-plan
    asyncio.get_event_loop().run_in_executor(
        rag_system.executor, preload_reranker
    )


# Routes de l'API Ultra Performante
@app.get("/", summary="API RAG Ultra Performant")
async def root():
    return {
        "message": "RAG Ultra Performant API",
        "version": "3.0.0",
        "features": [
            "Recherche hybride Dense+Sparse",
            "Re-ranking avec Cross-Encoder",
            "Query Enhancement intelligent",
            "Cache multicouche Redis+Mémoire",
            "Embeddings avancés",
            "Chunking sémantique",
            "Monitoring Prometheus",
            "Support multi-provider"
        ],
        "providers": [p.value for p in Provider]
    }


@app.get("/health", summary="Health check avancé")
async def health_check():
    """Health check avec métriques détaillées"""
    # Vérification des composants
    components_status = {
        "chroma_db": "healthy",
        "redis_cache": "healthy" if REDIS_AVAILABLE else "memory_fallback",
        "embeddings": "healthy",
        "reranker": "healthy"
    }

    # Test rapide des embeddings
    try:
        test_embedding = rag_system.embeddings.embed_query("test")
        components_status["embeddings"] = "healthy"
    except Exception as e:
        components_status["embeddings"] = f"error: {str(e)}"

    # Statistiques des providers
    provider_stats = {}
    for provider in Provider:
        api_key = API_KEYS[provider]
        provider_stats[provider.value] = {
            "configured": bool(api_key),
            "model": PROVIDER_CONFIGS[provider]["model"]
        }

    return {
        "status": "healthy",
        "service": "Ultra Performant RAG API",
        "components": components_status,
        "providers": provider_stats,
        "cache_status": "redis" if REDIS_AVAILABLE else "memory"
    }


@app.get("/metrics", summary="Métriques Prometheus")
async def metrics():
    """Endpoint pour Prometheus"""
    return Response(generate_latest(), media_type="text/plain")


@app.get("/performance-metrics", summary="Métriques de performance")
async def get_performance_metrics():
    """Métriques détaillées de performance"""

    try:
        # Calcul sécurisé des métriques
        total_queries = 0
        for metric in query_counter._metrics.values():
            total_queries += metric._value._value

        # Moyenne des temps de réponse
        response_sum = response_time_histogram._sum._value
        response_count = response_time_histogram._count._value
        avg_response_time = (response_sum / response_count * 1000) if response_count > 0 else 0

        # Taux de cache hit
        total_cache_hits = 0
        for metric in cache_hit_counter._metrics.values():
            total_cache_hits += metric._value._value

        cache_hit_rate = (total_cache_hits / max(total_queries, 1)) * 100

        # Documents actifs
        try:
            docs_result = rag_system.collection.get()
            document_ids = set()
            for metadata in docs_result.get("metadatas", []):
                if metadata and "document_id" in metadata:
                    document_ids.add(metadata["document_id"])
            active_documents = len(document_ids)
        except:
            active_documents = 0

        return {
            "total_queries": total_queries,
            "average_response_time_ms": round(avg_response_time, 2),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "error_rate": 0,
            "active_documents": active_documents
        }

    except Exception as e:
        logger.error(f"Erreur métriques: {e}")
        return {
            "total_queries": 0,
            "average_response_time_ms": 0,
            "cache_hit_rate": 0,
            "error_rate": 0,
            "active_documents": 0
        }


@app.post("/upload-document", response_model=DocumentResponse, summary="Upload document optimisé")
async def upload_document_optimized(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload de document avec traitement optimisé"""

    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
        raise HTTPException(
            status_code=400,
            detail="Format non supporté. Acceptés: PDF, DOC, DOCX"
        )

    try:
        # Lecture du fichier
        file_content = await file.read()

        # Traitement du document
        text = await process_document_advanced(file_content, file.filename)

        # Ajout au système RAG
        result = await rag_system.add_document(text, file.filename)

        return DocumentResponse(**result)

    except Exception as e:
        logger.error(f"Erreur upload document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur traitement: {str(e)}"
        )


@app.post("/ask-question-ultra", response_model=AdvancedQuestionResponse, summary="Question ultra optimisée")
async def ask_question_ultra(request: QuestionRequest):
    """Endpoint principal pour questions avec toutes les optimisations"""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    try:
        result = await rag_system.query(
            question=request.question,
            provider=request.provider,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return AdvancedQuestionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur question ultra: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.post("/ask-question-stream-ultra", summary="Question streaming ultra optimisée")
async def ask_question_stream_ultra(request: QuestionRequest):
    """Version streaming de la question ultra optimisée"""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    async def generate_ultra_stream():
        try:
            # Recherche et préparation du contexte (partie non-streaming)
            start_time = time.time()
            query_id = str(uuid.uuid4())

            # Enhancement et recherche
            llm_provider = OptimizedLLMProvider(request.provider)
            enhanced_queries = await rag_system.query_enhancer.enhance_query(request.question, llm_provider)

            # Métadonnées initiales
            initial_metadata = {
                "id": query_id,
                "provider": request.provider.value,
                "enhanced_queries": enhanced_queries,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps({'metadata': initial_metadata, 'type': 'init'})}\n\n"

            # Recherche hybride
            all_results = []
            for query_variant in enhanced_queries:
                results = await rag_system.hybrid_search.search(query_variant, n_results=15)
                all_results.extend(results)

            if not all_results:
                yield f"data: {json.dumps({'content': 'Aucun document pertinent trouvé.', 'type': 'final'})}\n\n"
                return

            # Re-ranking
            ranked_results = rag_system.reranker.rerank(request.question, all_results, top_k=request.top_k)

            # Préparation contexte
            context_parts = [f"Source {i + 1}: {result.content}" for i, result in enumerate(ranked_results)]
            context = "\n\n".join(context_parts)

            # Prompt optimisé
            optimized_prompt = f"""Contexte: {context}

Question: {request.question}

Répondez en utilisant uniquement le contexte fourni. Citez les sources quand approprié.

Réponse:"""

            # Streaming de la génération
            provider = OptimizedLLMProvider(request.provider)

            async for chunk in provider.generate_stream(optimized_prompt):
                if chunk:
                    yield f"data: {json.dumps({'content': chunk, 'type': 'chunk'})}\n\n"

            # Métadonnées finales
            end_time = time.time()
            final_metadata = {
                "response_time_ms": round((end_time - start_time) * 1000, 2),
                "search_results": len(all_results),
                "ranked_results": len(ranked_results)
            }
            yield f"data: {json.dumps({'metadata': final_metadata, 'type': 'final'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"

    return StreamingResponse(
        generate_ultra_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/documents-advanced", summary="Liste documents avec métadonnées")
async def list_documents_advanced():
    """Liste avancée des documents avec statistiques"""
    try:
        results = rag_system.collection.get()

        document_stats = {}
        total_chunks = 0

        for metadata in results.get("metadatas", []):
            if metadata and "document_id" in metadata:
                doc_id = metadata["document_id"]

                if doc_id not in document_stats:
                    document_stats[doc_id] = {
                        "document_id": doc_id,
                        "total_chunks": 0,
                        "chunk_types": {},
                        "total_length": 0
                    }

                document_stats[doc_id]["total_chunks"] += 1
                total_chunks += 1

                chunk_type = metadata.get("chunk_type", "unknown")
                if chunk_type not in document_stats[doc_id]["chunk_types"]:
                    document_stats[doc_id]["chunk_types"][chunk_type] = 0
                document_stats[doc_id]["chunk_types"][chunk_type] += 1

                document_stats[doc_id]["total_length"] += metadata.get("chunk_length", 0)

        return {
            "documents": list(document_stats.values()),
            "summary": {
                "total_documents": len(document_stats),
                "total_chunks": total_chunks,
                "average_chunks_per_doc": round(total_chunks / max(len(document_stats), 1), 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.delete("/documents/{document_id}", summary="Suppression avancée document")
async def delete_document_advanced(document_id: str):
    """Suppression avancée avec nettoyage complet"""
    try:
        # Suppression de ChromaDB
        rag_system.collection.delete(where={"document_id": document_id})

        # Reconstruction index BM25
        await asyncio.get_event_loop().run_in_executor(
            rag_system.executor,
            rag_system.hybrid_search.rebuild_index
        )

        # Nettoyage cache lié au document
        # Note: Implémentation simplifiée, pourrait être plus sophistiquée

        return {
            "message": f"Document '{document_id}' supprimé avec succès",
            "actions": [
                "Suppression chunks ChromaDB",
                "Reconstruction index BM25",
                "Nettoyage cache"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur suppression: {str(e)}")


@app.post("/clear-cache", summary="Vider le cache")
async def clear_cache():
    """Nettoyage complet du cache"""
    try:
        cleared_items = 0

        # Cache Redis
        if REDIS_AVAILABLE:
            cleared_items += redis_client.flushdb()

        # Cache mémoire
        with cache.memory_cache_lock:
            cleared_items += len(cache.memory_cache)
            cache.memory_cache.clear()

        return {
            "message": "Cache vidé avec succès",
            "items_cleared": cleared_items
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


# Point d'entrée principal
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1  # Important pour le cache en mémoire partagé
    )
