from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from elasticsearch import Elasticsearch


# Initialisation Elasticsearch

es = Elasticsearch()


# Indexation des documents

def index_document(doc_id, text):
    es.index(index='docs', id=doc_id, body={'text': text})


# Recherche de documents
def search_docs(query):
    response = es.search(index='docs', body={'query': {'match': {'text': query}}})
    return [hit['_source']['text'] for hit in response['hits']['hits']]


    # Initialisation du modèle RAG
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)


# Génération de réponse

def generate_response(query):
    inputs_dict = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")
    generated_ids = model.generate(**inputs_dict)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


# Boucle principale

while True:
    query = input("Entrez votre question : ")
    docs = search_docs(query)
    response = generate_response(query)
    print("Réponse :", response)
