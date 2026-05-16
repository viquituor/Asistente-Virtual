import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageEmbedder, ImageEmbedderOptions, RunningMode

import chromadb as db

MODELO = "modelos/mobilenet_v3_large.tflite"

NOME_BANCO = "bichos"
CAMINHO_BANCO = "banco"

GATOS_BRANCOS = [
    "imagens/gato_branco1.png",
    "imagens/gato_branco2.png"
]

GATOS_PRETOS = [
    "imagens/gato_preto1.png",
    "imagens/gato_preto2.png"
]

CACHORROS = [
    "imagens/cachorro1.png",
    "imagens/cachorro2.png"
]


def iniciar():
    iniciado, embedador, conexao_bd = False, None, None
    
    try:

        configuracoes = ImageEmbedderOptions(base_options=BaseOptions(model_asset_path=MODELO), running_mode=RunningMode.IMAGE)
        embedador = ImageEmbedder.create_from_options(configuracoes)
        conexao_bd = db.PersistentClient(path=CAMINHO_BANCO)

        iniciado = True
    except Exception as e:
        print(f"ocorreu um erro iniciando o embedador ou a conexão com o banco de dados: {e}")
    
    return iniciado, embedador, conexao_bd


def processar(imagem, embedador):
    processado, embedding = False, None
    
    try:

        imagem = mp.Image.create_from_file(imagem)
        embedding = embedador.embed(imagem)

        processado = True
    except Exception as e:
        print(f"ocorreu um erro processando a imagem: {e}")

    return processado, embedding


def processar_bichos(imagens, embedador):
    processados, embeddings = False, []

    for imagem in imagens:
        processado, embedding = processar(imagem, embedador)
        if processado:
            embeddings.append(embedding)

    processados = len(embeddings) == len(imagens)

    return processados, embeddings


def converter_embedding(embedding):
    conversao = []

    for valor in embedding:
        conversao.append(int(valor))

    return conversao


def gravar_embeddings(tipo_bicho, embedding, conexao_bd):
    gravados = False
    
    try:
        colecao = conexao_bd.get_or_create_collection(name=NOME_BANCO)
        conversao = converter_embedding(embedding.embeddings[0].embedding)
        
        for id, embedding in enumerate(embeddings):
            colecao.add(embeddings=[conversao], metadatas=[{"bicho":tipo_bicho}], ids=[f"{tipo_bicho}_{str(id+1)}"])

        gravados = True 
    except Exception as e:
        print(f"ocorreu um erro gravando os embeddings no banco de dados: {e}")
        
    return gravados

if __name__ == "__main__":
    iniciado, embedador, conexao_bd = iniciar()
    if iniciado:
        processados, embeddings = processar_bichos(GATOS_BRANCOS, embedador)
        if processados:
            gravados = gravar_embeddings("gatos brancos", embeddings, conexao_bd)
            if gravados:
                print("embeddings de gatos brancos gravados com sucesso!")
            
        processados, embeddings = processar_bichos(GATOS_PRETOS, embedador)
        if processados:
            gravados = gravar_embeddings("gatos pretos", embeddings, conexao_bd)
            if gravados:
                print("embeddings de gatos pretos gravados com sucesso!")
                
        processados, embeddings = processar_bichos(CACHORROS, embedador)
        if processados:
            gravados = gravar_embeddings("cachorros", embeddings, conexao_bd)
            if gravados:
                print("embeddings de cachorros gravados com sucesso!")