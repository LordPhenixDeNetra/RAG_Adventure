# RAG Ultra Performant API

## 📋 Description

Ce projet est une API RAG (Retrieval Augmented Generation) ultra-performante construite avec FastAPI. Elle intègre des techniques avancées de recherche hybride, de re-ranking et d'optimisation pour fournir des réponses précises et rapides à partir de documents.

## ✨ Fonctionnalités

- 🔍 **Recherche Hybride** : Combinaison de recherche dense (vectorielle) et sparse (BM25)
- 🏆 **Re-ranking** : Utilisation de Cross-Encoder pour améliorer la pertinence des résultats
- 💡 **Query Enhancement** : Génération automatique de variantes de requêtes
- 🗄️ **Cache Multicouche** : Redis + mémoire pour une performance optimale
- 📊 **Monitoring** : Métriques Prometheus intégrées
- 🤖 **Multi-providers** : Support de plusieurs providers LLM (Mistral, OpenAI, Anthropic, Deepseek, Groq)
- ⚡ **Streaming** : Réponses en temps réel
- 🔧 **Gestion avancée des documents** : Chunking sémantique et métadonnées enrichies

## 🛠️ Installation

### Prérequis

- Python 3.8+
- Redis (optionnel mais recommandé)
- Clés API pour les providers LLM souhaités

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Variables d'environnement

Créez un fichier `.env` à la racine du projet :

```env
# Clés API (selon les providers utilisés)
MISTRAL_API_KEY=votre_cle_mistral
OPENAI_API_KEY=votre_cle_openai
ANTHROPIC_API_KEY=votre_cle_anthropic
DEEPSEEK_API_KEY=votre_cle_deepseek
GROQ_API_KEY=votre_cle_groq

# Configuration Redis (optionnel)
REDIS_HOST=localhost
REDIS_PORT=6379

# Autres paramètres
EMBEDDINGS_MODEL=all-mpnet-base-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
```

## 🚀 Démarrage

### Développement

```bash
python main.py
```

### Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

L'API sera accessible à l'adresse : `http://localhost:8000`

## 📚 API Endpoints

### 📄 Documents

- `POST /upload-document` - Téléverser un document (PDF, DOC, DOCX)
- `GET /documents-advanced` - Lister les documents avec métadonnées
- `DELETE /documents/{document_id}` - Supprimer un document

### ❓ Questions

- `POST /ask-question-ultra` - Poser une question (mode standard)
- `POST /ask-question-stream-ultra` - Poser une question (mode streaming)

### 📊 Monitoring

- `GET /health` - Statut de santé de l'API
- `GET /metrics` - Métriques Prometheus
- `GET /performance-metrics` - Métriques de performance détaillées
- `POST /clear-cache` - Vider le cache

## 🔧 Configuration avancée

### Providers LLM supportés

| Provider | Modèle par défaut | Statut |
|----------|-------------------|--------|
| Mistral | mistral-medium | ✅ |
| OpenAI | gpt-4o-mini | ✅ |
| Anthropic | claude-3-haiku-20240307 | ✅ |
| Deepseek | deepseek-chat | ✅ |
| Groq | mixtral-8x7b-32768 | ✅ |

### Paramètres de recherche

- `top_k`: Nombre de résultats à retourner (défaut: 3)
- `temperature`: Contrôle la créativité des réponses (défaut: 0.3)
- `max_tokens`: Longueur maximale des réponses (défaut: 512)

## 🧩 Architecture technique

### Composants principaux

1. **AdvancedEmbeddings** - Gestion des embeddings avec cache
2. **AdvancedChunker** - Chunking sémantique intelligent
3. **HybridSearch** - Recherche hybride dense + sparse
4. **AdvancedReranker** - Re-ranking avec Cross-Encoder
5. **QueryEnhancer** - Amélioration des requêtes
6. **OptimizedLLMProvider** - Abstraction multi-provider
7. **MultiLayerCache** - Cache multicouche Redis + mémoire

### Flux de traitement

1. Téléversement → Chunking → Stockage vectoriel
2. Requête → Enhancement → Recherche hybride → Re-ranking → Génération
3. Réponse → Cache → Métriques

## 📊 Métriques et monitoring

L'API expose des métriques Prometheus pour :

- Temps de réponse
- Taux de réussite des requêtes
- Utilisation du cache
- Performance des providers
- Nombre de documents et chunks

## 🔍 Exemples d'utilisation

### Téléverser un document

```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload-document
```

### Poser une question

```bash
# Utilisation de Mistral (par défaut)
curl -X POST -H "Content-Type: application/json" -d '{
  "question": "Quelle est la conclusion principale du document?",
  "provider": "mistral",
  "top_k": 3
}' http://localhost:8000/ask-question-ultra

# Utilisation d'OpenAI
curl -X POST -H "Content-Type: application/json" -d '{
  "question": "Quelle est la conclusion principale du document?",
  "provider": "openai",
  "top_k": 3
}' http://localhost:8000/ask-question-ultra

# Utilisation d'Anthropic
curl -X POST -H "Content-Type: application/json" -d '{
  "question": "Quelle est la conclusion principale du document?",
  "provider": "anthropic",
  "top_k": 3
}' http://localhost:8000/ask-question-ultra

# Utilisation de Deepseek
curl -X POST -H "Content-Type: application/json" -d '{
  "question": "Quelle est la conclusion principale du document?",
  "provider": "deepseek",
  "top_k": 3
}' http://localhost:8000/ask-question-ultra

# Utilisation de Groq
curl -X POST -H "Content-Type: application/json" -d '{
  "question": "Quelle est la conclusion principale du document?",
  "provider": "groq",
  "top_k": 3
}' http://localhost:8000/ask-question-ultra


```

### Obtenir les métriques

```bash
curl http://localhost:8000/metrics
```

## 🚦 Performance

### Optimisations

- Cache multicouche (Redis + mémoire)
- Pré-chargement des modèles
- Traitement par batch
- Pool de threads pour opérations I/O
- Embedding caching avec LRU
- Indexation BM25 optimisée

### Métriques clés

- Temps de réponse moyen : < 500ms
- Support de gros volumes de documents
- Cache hit rate > 70% avec Redis
- Faible consommation mémoire

## 🐛 Dépannage

### Problèmes courants

1. **Redis non disponible** : Le système bascule sur le cache mémoire
2. **Provider API inaccessible** : Vérifiez les clés API et la connectivité
3. **Documents non trouvés** : Vérifiez le format des fichiers (PDF/DOC/DOCX)

### Logs

Les logs détaillés sont disponibles avec différents niveaux de verbosité :

```python
logging.basicConfig(level=logging.INFO)  # ou DEBUG pour plus de détails
```

## 🤝 Contribuer

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créez une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🏷️ Versions

- **v3.0.0** - Version ultra-performante avec recherche hybride et re-ranking
- **v2.0.0** - Support multi-provider et caching avancé
- **v1.0.0** - Version initiale avec fonctionnalités de base RAG

## 📞 Support

Pour toute question ou problème, veuillez ouvrir une issue sur le repository GitHub.

---

**Note** : Cette API est conçue pour des environnements de production et inclut des fonctionnalités avancées de monitoring, caching et gestion d'erreurs.