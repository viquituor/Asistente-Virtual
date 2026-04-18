def iniciar():
    print("tocador iniciado")

def atuar(acao, objeto, local):
    if acao in ["tocar"] and objeto in ["música", "som"]:
        print(f"tocando uma música")
    elif acao in ["parar"] and objeto in ["música", "som"]:
        print(f"parando a música")
    else:
        print("comando não reconhecido pelo tocador")

        