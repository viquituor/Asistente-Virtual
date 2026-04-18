import unittest
from assistente import *

class TestesLampada(unittest.TestCase):
    @classmethod

    def setUpClass(cls):
        dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"

        cls.iniciado, cls.processador, cls.modelo, cls.palavras_de_parada, cls.acoes = iniciar_assistente(dispositivo)

        return super().setUpClass()
    
    def testar_01_assistente_iniciado(self):
        self.assertTrue(self.iniciado)


unittest.main()