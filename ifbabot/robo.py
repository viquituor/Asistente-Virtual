from chatterbot import ChatBot
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NOME_ROBO = "IFBA.Bot"
LIMIAR_ACEITACAO = 0.6

import time
time.clock = time.time

def iniciar():
    iniciado, robo = False, None

    try:
        robo = ChatBot(NOME_ROBO)

        iniciado  = True 
    except Exception as e:
        print(f"erro iniciando robô: {e}")

    return iniciado, robo

def get_resposta(robo, mensagem):
    resposta = robo.get_response(mensagem.lower())


    return resposta.confidence, resposta.text

if __name__ == "__main__":
    iniciado, robo = iniciar()
    if iniciado:
        while True:
            
            mensagem = input("")
            confianca, resposta = get_resposta(robo, mensagem)
            if confianca >= LIMIAR_ACEITACAO:
                print(f"🤖: {resposta} (confiança = {confianca})")
            else:
                print("🤖: Desculpe, não entendi o que você quis dizer. Poderia reformular a pergunta?")