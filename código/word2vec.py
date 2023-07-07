import numpy as np
import gensim


class Representation():
    '''Transformar as palavras em vetores (representação)'''

    def __init__(self, num_dimensoes, coluna_arquivo):
       self.num_dimensoes = num_dimensoes
       self.coluna = coluna_arquivo
    

    def gerar_representacao(self, metodo='sum'):
        '''
        Args:
            texto (String): 
            metodo (String): 

        return:
        
        '''
        matrix = np.zeros((len(self.coluna), self.num_dimensoes))
        for i in range(len(self.coluna)):
            tokens = self.coluna.iloc[i]
            matrix[i] = self._soma_vetores(tokens)
            if metodo == 'average' and len(tokens) > 0:
                matrix[i] = matrix[i]/len(tokens)
        return matrix


    def _soma_vetores(self, tokens):
        '''
        Args:
            tokens (String):  

        return:
        
        '''
        #Criando o modelo da linguagem
        modelo_linguagem = gensim.models.Word2Vec(self.coluna,sg=0,min_count=1,window=10,vector_size=self.num_dimensoes)
        vetor_texto = np.zeros(self.num_dimensoes)
        for token in tokens:
            try:
                vetor_texto += modelo_linguagem.wv.get_vector(token)
            except KeyError:
                continue
        return vetor_texto
    