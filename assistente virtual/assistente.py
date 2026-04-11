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
import torch        # Importação em falta
import torchaudio   # Importação em falta

CONFIGURACAO = "assistente virtual\config.json"
LINGUAGEM = "portuguese" # Corrigido de LINUAGEM para LINGUAGEM
TEMPO_GRAVACAO = 8
CAMINHO_AUDIO_FALAS = "temp"



def iniciar_assistente(dispositivo):
    iniciado, processador, modelo = iniciar_modelo(MODELO, dispositivo)

    palavras_de_parada = set(corpus.stopwords.words(LINGUAGEM))

    with open(CONFIGURACAO, "r", encoding="utf-8") as arquivo_configuracao:
        configuracoes = json.load(arquivo_configuracao)
        acoes = configuracoes["acoes"]

        arquivo_configuracao.close()

    return iniciado, processador, modelo, palavras_de_parada

def capturar_fala():
    print("fale algo")
    fala = sd.rec(int(TEMPO_GRAVACAO * TAXA_AMOSTRAGEM), samplerate=TAXA_AMOSTRAGEM, channels=1)
    sd.wait()
    print("gravacao finalizada")
    return fala

def gravar_fala(fala):
    # Garante que a pasta temporária existe antes de gravar
    os.makedirs(CAMINHO_AUDIO_FALAS, exist_ok=True)

    gravado, arquivo = False, f"{CAMINHO_AUDIO_FALAS}/{secrets.token_hex(32)}.wav"

    try:
        sf.write(arquivo, fala, TAXA_AMOSTRAGEM)
        gravado = True
    except Exception as e:
        print(f"erro gravando fala: {e}")

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
        acao = comando[0]
        objeto = comando[1]
        local = comando[2]

        for acao_configurada in acoes:
            if acao == acao_configurada["nome"]:
                if objeto in acao_configurada["dispositivos"]:
                    valido = True

                    break
    
    return valido, acao, objeto, local

def executar_acao(acao, objeto, local):
    print(f"executando acao: {acao} no objeto: {objeto} no local: {local}")
    # Aqui você pode adicionar a lógica para controlar os dispositivos reais, como enviar comandos para um sistema de automação residencial, etc.


if __name__ == "__main__":

    dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"

    iniciado, processador, modelo, palavras_de_parada, acoes = iniciar_assistente(dispositivo)
    if iniciado:
        fala = capturar_fala()
        gravado, arquivo = gravar_fala(fala)
        if gravado:
            print("AGUARDE... realizando transcricao da fala...")
            fala, _ = torchaudio.load(arquivo)

            transcricao = transcrever(dispositivo, fala.squeeze(), modelo, processador)

            print(f"fala: {transcricao}")

            comando = processar_transcricao(transcricao, palavras_de_parada)
            print(f"comando: {comando}")

            valido,acao,objeto, local = validar_comando(comando, acoes)
            if valido:
                executar_acao(acao, objeto, local)
            else:
                print("Comando Invalido")
    else:
        print("não foi possivel iniciar a gravação")