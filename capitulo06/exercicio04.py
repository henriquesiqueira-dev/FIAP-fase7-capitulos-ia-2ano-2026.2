from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from google.colab import userdata

os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

# Criar embeddings e indexar textos
embeddings = OpenAIEmbeddings()
textos = ["Documentação de produto A", "Manual de operação B", "Relatório técnico C"]
vector_db = Chroma.from_texts(texts=textos, embedding=embeddings)

# Consulta de exemplos
consulta = "Como configurar o produto A?"
resultados = vector_db.similarity_search(consulta, k=1)
for r in resultados:
    print("Trecho relevante:", r.page_content)
