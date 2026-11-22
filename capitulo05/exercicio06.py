import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util

# 1) EMBEDDINGS
modelo = SentenceTransformer('all-MiniLM-L6-v2')

documentos = [
    "Este manual explica como instalar o software.",
    "Guia de operação do sistema.",
    "Especificações técnicas do produto."
]

embeddings_docs = modelo.encode(documentos, convert_to_tensor=True)
consulta = "Quais são os passos para instalar?"
embedding_consulta = modelo.encode(consulta, convert_to_tensor=True)

# Cosine similarities (query vs docs)
similaridades = util.cos_sim(embedding_consulta, embeddings_docs).cpu().numpy().flatten()

indice_mais_similar = int(np.argmax(similaridades))
print("Documento mais relevante:", documentos[indice_mais_similar])
print("Similaridades (cos):", {i: float(similaridades[i]) for i in range(len(documentos))})

# 2) PCA (2D) MANUAL (SVD)
# Build matrix with query + docs
X = np.vstack([
    embedding_consulta.cpu().numpy(),
    embeddings_docs.cpu().numpy()
]).astype(np.float64)

labels = ["Consulta"] + [f"Doc {i}" for i in range(len(documentos))]

# Center
X_centered = X - X.mean(axis=0, keepdims=True)

# SVD -> principal components
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
PC = Vt[:2].T  # (d, 2)
X_2d = X_centered @ PC  # (n, 2)

# 3) DISTANCES IN PCA SPACE
q_xy = X_2d[0]
docs_xy = X_2d[1:]

# Euclidean distances in PCA space (query -> each doc)
dist_pca = np.linalg.norm(docs_xy - q_xy, axis=1)

# For reference: cosine distance in original space (1 - cosine similarity)
dist_cos = 1.0 - similaridades

print("Distâncias (PCA euclidiana) query->docs:", {i: float(dist_pca[i]) for i in range(len(documentos))})
print("Distâncias (cos) query->docs:", {i: float(dist_cos[i]) for i in range(len(documentos))})

# 4) PLOTS
# Scatter plot in PCA space
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[0, 0], X_2d[0, 1], marker="*", s=250, label="Consulta")
for i in range(len(documentos)):
    plt.scatter(X_2d[i+1, 0], X_2d[i+1, 1], s=120, label=f"Doc {i}")
    # line from query to doc
    plt.plot([X_2d[0, 0], X_2d[i+1, 0]], [X_2d[0, 1], X_2d[i+1, 1]])
    # distance annotation
    midx = (X_2d[0, 0] + X_2d[i+1, 0]) / 2
    midy = (X_2d[0, 1] + X_2d[i+1, 1]) / 2
    plt.text(midx, midy, f"{dist_pca[i]:.3f}")

plt.title("PCA (2D) — Consulta vs Documentos (linhas = distâncias PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# Bar chart of distances (PCA space)
plt.figure(figsize=(8, 4))
plt.bar([f"Doc {i}" for i in range(len(documentos))], dist_pca)
plt.title("Distância Euclidiana no Espaço PCA (Consulta → Documento)")
plt.xlabel("Documento")
plt.ylabel("Distância (PCA)")
plt.grid(True, axis="y")
plt.show()

# Bar chart of cosine distance (original embedding space)
plt.figure(figsize=(8, 4))
plt.bar([f"Doc {i}" for i in range(len(documentos))], dist_cos)
plt.title("Distância Cosseno no Espaço Original (Consulta → Documento)")
plt.xlabel("Documento")
plt.ylabel("Distância (1 - cos_sim)")
plt.grid(True, axis="y")
plt.show()
