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

LINUAGEM = "portuguese"
TEMPO_GRAVACAO = 5
CAMINHO_AUDIO_FALAS = "temp"

def iniciar_assistente(dispositivo):
    iniciado, processador, modelo= iniciar_modelo(MODELO, dispositivo)
    
    palavras_de_parada = set(corpus.stopwords.words(LINUAGEM))
    
    return iniciado, processador, modelo, palavras_de_parada
    

def capturar_fala():
    print("fale algo")
    fala = sd.rec(int(TEMPO_GRAVACAO * TAXA_AMOSTRAGEM), samplerate=TAXA_AMOSTRAGEM, channels=1)
    sd.wait()
    print("gravação finalizada")
    return fala

def gravar_fala(fala):
    gravado, arquivo = False, f"{CAMINHO_AUDIO_FALAS}/{secrets.token_hex(32)}.wav"
    
    try:
        sf.write(arquivo, fala, TAXA_AMOSTRAGEM)
        gravado = True
    except Exception as e:
        print(f"erro gravando fala: {e}")
        
    return gravado, arquivo

if __name__ == "__main__":
    
    dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    iniciado, processador, modelo, palavras_de_parada = iniciar_assistente(dispositivo)
    if iniciado:
        fala = capturar_fala()
        gravado,arquivo = gravar_fala(fala)
        if gravado:
            print("realizando gravacao da fala...")
            fala, _ = torchaudio.load(arquivo)
            transcrever(dispositivo, fala.squeeze(), modelo, processador)
                
            print(f"fala: {transcricao}")
    else:
             print("nap foi possivel iniciar a gravaçao") 