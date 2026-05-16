from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import json

NOME_ROBO = "IFBA.Bot"


def iniciar():
    iniciado, robo = False, None
    
    try:
        robo = ChatBot(NOME_ROBO)
        treinador = ListTrainer(robo)
        
        iniciado  = True 
    except Exception as e:
        print(f"erro iniciando robô: {e}")
    
    return iniciado, robo, treinador

def carregar_conversas():
    ...
    
def treinar(treinador, conversas):
    ...
    
if __name__ == "__main__":
    iniciado, robo, treinador = iniciar()
    if iniciado:
        carregadas, conversas = carregar_conversas()
        if carregadas:
            treinar(treinador, conversas)