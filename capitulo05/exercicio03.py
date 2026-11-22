from openai import OpenAI
from google.colab import userdata
import os

os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

# Crie o cliente com sua chave
client = OpenAI()

pergunta = "Quem é o melhor jogador do mundo? Responda apenas o nome do jogador"

# Chamada ao modelo (ChatGPT)
response = client.chat.completions.create(
    model="gpt-4o-mini",  # modelo leve e econômico
    messages=[

        {"role": "user", "content": pergunta}
    ],
    max_tokens=100,
    temperature=0
)

print("Resposta do LLM:", response.choices[0].message.content)
