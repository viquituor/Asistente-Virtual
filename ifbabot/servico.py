from robo import *
from flask import Flask
import json

iniciado, robo = iniciar()

servico = Flask(NOME_ROBO)

@servico.get("/")
def get_info():
    return json.dumps({
        "nome": NOME_ROBO,
        "descricao": "IFBA.Bot é um chatbot desenvolvido para responder perguntas sobre o Instituto Federal da Bahia (IFBA). Ele é treinado com informações básicas sobre o IFBA, saudações e sistemas de informação. O objetivo do IFBA.Bot é fornecer respostas rápidas e precisas para os usuários que desejam saber mais sobre o IFBA."
        })
    
if __name__ == "__main__":
    