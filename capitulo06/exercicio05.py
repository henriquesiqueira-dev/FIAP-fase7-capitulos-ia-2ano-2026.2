# Fazendo a INSTALAÇÃO:
!pip install faiss-cpu sentence-transformers openai



# 1) IMPORTS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os
from google.colab import userdata

os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

# 2) EMBEDDING MODEL

modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Base de conhecimento
textos = [
    "Manual de usuário: Para instalar o produto, execute o instalador e siga as instruções na tela.",
    "Guia de instalação: O produto requer Python 3.10 ou superior.",
    "Especificações técnicas: O sistema suporta Windows e Linux."
]

embeddings = modelo.encode(textos)

# 3) FAISS INDEX (VectorDB)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# 4) CONSULTA (retrieval)
consulta = "Como instalar o produto?"
consulta_emb = modelo.encode([consulta])

distancias, indices = index.search(np.array(consulta_emb), k=2)

contextos_recuperados = [textos[i] for i in indices[0]]

print("Trechos recuperados:")
for trecho in contextos_recuperados:
    print("-", trecho)

# 5) OPENAI GENERATOR (GPT-4o-mini)
client = OpenAI()

# Construção do prompt RAG
prompt = f"""
Use exclusivamente o contexto abaixo para responder à pergunta.

Contexto:
{chr(10).join(contextos_recuperados)}

Pergunta:
{consulta}
"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt,
    max_output_tokens=200,
    temperature=0.2
)

print("\nResposta final do modelo:")
print(response.output_text)
