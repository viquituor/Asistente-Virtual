import pygame
MUSICA = "Asistente-Virtual\\assistente virtual\\audios\\la belle de jour.mp3"

def iniciar():
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(MUSICA)
    print("tocador iniciado")

def atuar(acao, objeto, local):
    if acao in ["tocar"] and objeto in ["música", "som"]:
        print(f"tocando uma música")
    
        pygame.mixer.music.play()
    elif acao in ["parar"] and objeto in ["música", "som"]:
        print(f"parando a música")

        pygame.mixer.music.stop()
    else:
        print("comando não reconhecido pelo tocador")

        