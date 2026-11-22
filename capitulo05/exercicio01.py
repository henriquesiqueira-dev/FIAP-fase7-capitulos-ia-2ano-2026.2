# Exemplo mínimo de pipeline RAG com Python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Carrega modelo de embeddings
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

# Documentos de exemplo
docs = [
    "O Brasil tem 26 estados e 1 distrito federal.",
    "São Paulo é o estado mais populoso do Brasil.",
    "A capital do Brasil é Brasília."
]

# Gera embeddings
doc_embeddings = emb_model.encode(docs)

# Exemplo de consulta
query = "Qual é a capital do Brasil?"
query_emb = emb_model.encode([query])

# Recupera índice do documento mais similar
similarities = cosine_similarity(query_emb, doc_embeddings)
best_idx = np.argmax(similarities)

print(f"Documento mais relevante: {docs[best_idx]}")
