from flask import Flask, Response, request, send_from_directory
from nltk import word_tokenize, corpus
from inicializador_modelo import *
from threading import Thread
from transcritor import *
import sounddevice as sd
import soundfile as sf
import secrets
import json
import os

import lampada
import tocador

CONFIGURACAO = "Asistente-Virtual\\assistente virtual\\config.json"

LINGUAGEM = "portuguese"
TEMPO_GRAVACAO = 5
CAMINHO_AUDIO_FALAS = "temp"

ATUADORES = [
    {
        "nome": "lâmpada",
        "iniciar": lampada.iniciar,
        "atuar": lampada.atuar
    },
    {
        "nome": "tocador",
        "iniciar": tocador.iniciar,
        "atuar": tocador.atuar
    }
]

def iniciar_assistente(dispositivo):
    iniciado, processador, modelo = iniciar_modelo(MODELO, dispositivo)

    palavras_de_parada = set(corpus.stopwords.words(LINGUAGEM))

    with open(CONFIGURACAO, "r", encoding="utf-8") as arquivo_configuracao:
        configuracoes = json.load(arquivo_configuracao)
        acoes = configuracoes["acoes"]

        arquivo_configuracao.close()

    for atuador in ATUADORES:
        atuador["iniciar"]()

    return iniciado, processador, modelo, palavras_de_parada, acoes

def capturar_fala():
    print("fale alguma coisa...")

    fala = sd.rec(int(TEMPO_GRAVACAO * TAXA_AMOSTRAGEM), samplerate=TAXA_AMOSTRAGEM, channels=1)
    sd.wait()

    print("fala capturada!")

    return fala

def gravar_fala(fala):
    gravado, arquivo = False, f"{CAMINHO_AUDIO_FALAS}/{secrets.token_hex(32).lower()}.wav"

    try:
        sf.write(arquivo, fala, TAXA_AMOSTRAGEM)

        gravado = True
    except Exception as e:
        print(f"ocorreu um erro gravando o áudio: {e}")

    return gravado, arquivo

def processar_transcricao(transcricao, palavras_de_parada):
    tokens = word_tokenize(transcricao)

    comando = []
    for token in tokens:
        if token not in palavras_de_parada:
            comando.append(token)

    return comando

def validar_comando(comando, acoes):
    valido, acao, objeto, local = False, None, None, None

    if len(comando) == 3:
        acao = comando[0] # ligar, desligar
        objeto = comando[1] # lâmpada, ventilador
        local = comando[2] # sala, cozinha, quarto

        for acao_configurada in acoes:
            if acao == acao_configurada["nome"]: # ligar
                if objeto in acao_configurada["dispositivos"]:
                    valido = True

                    break

    return valido, acao, objeto, local

def executar_comando(acao, objeto, local):
    for atuador in ATUADORES:
        atuacao = Thread(target= atuador["atuar"], args=[acao,objeto,local]) 
        atuacao.start()

if __name__ == "__main__":
    dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"

    iniciado, processador, modelo, palavras_de_parada, acoes = iniciar_assistente(dispositivo)
    if iniciado:
        while True:
            fala = capturar_fala()
            gravado, arquivo = gravar_fala(fala)
            if gravado:
                print("realizando transcrição...")

                fala, _ = torchaudio.load(arquivo)
                transcricao = transcrever(dispositivo, fala.squeeze(), modelo, processador)
                print(f"fala: {transcricao}")

                comando = processar_transcricao(transcricao, palavras_de_parada)
                print(f"comando: {comando}")

                valido, acao, objeto, local = validar_comando(comando, acoes)
                if valido:
                    executar_comando(acao, objeto, local)
                else:
                    print("comando inválido")
    else:
        print("não possível iniciar o assistente")