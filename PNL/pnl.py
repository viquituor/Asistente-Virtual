from nltk import word_tokenize, corpus
from nltk.corpus import floresta
from nltk.stem import RSLPStemmer

linguagem = 'portuguese'
texto = "a verdadeira generosidade para com o futuro consiste em dar duro ao presente"

def iniciar():
    palavras_de_parada= set(corpus.stopwords.words(linguagem))
    
    classificacoes = {}
    for expressao, classificacao in floresta.tagged_words():
        classificacoes[expressao] = classificacao
        
    return palavras_de_parada, classificacoes

def tokenizar(texto):
    tokens = word_tokenize(texto)
    
    return tokens

def imprimir(tokens):
    for token in tokens:
        print(token)
        
def eliminar_palavras_de_parada(tokens, palavras_de_parada):
    tokens_filtrados = [token for token in tokens if token not in palavras_de_parada]
    
    return tokens_filtrados

def classificar_gramaticalmente(tokens, classificacoes):
    tokens_classificados = {}
    for token in tokens:
        classificacao = classificacoes[token]
        if classificacao == None:
            classificacao = 'Desconhecida'
        
        tokens_classificados[token] = classificacao
    return tokens_classificados

def estematizar(tokens):
    raizes_de_tokens = {}
    estemizador = RSLPStemmer()
    
    tokens_stemmatizados = [estemizador.stem(token) for token in tokens]
    
    return tokens_stemmatizados

if __name__ == "__main__":
    palavras_de_parada, classificacoes = iniciar()
    
    tokens = tokenizar(texto)
    
    #imprimir(tokens)
    
    tokens = eliminar_palavras_de_parada(tokens, palavras_de_parada)
    
    #imprimir(tokens)
    
    classificacoes_tokens = classificar_gramaticalmente(tokens, classificacoes)
    
    #for token, classificacao in classificacoes_tokens.items():
        #print(f"{+token}: {classificacao}")
        
    raizes_de_tokens = estematizar(tokens)
    
    print(raizes_de_tokens)
    