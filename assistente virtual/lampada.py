def iniciar():
    print("lâmpada iniciada")

def atuar(acao, objeto, local):
    if acao in ["ligar", "acender"] and objeto == "lâmpada":
        print(f"ligando a lâmpada no/na/em {local}")
    elif acao in ["desligar", "apagar"] and objeto == "lâmpada":
        print(f"desligando a lâmpada no/na/em {local}")
    else:
        print("comando não reconhecido pela lâmpada")